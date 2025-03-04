import time
from typing import List
import networkx as nx
import z3
import re
from tqdm import tqdm
import os
import logging
import subprocess
import pickle
from functools import partial
from kmax.arch import Arch
from kmax.klocalizer import Klocalizer
import multiprocessing as mp
mp.set_start_method("fork", force=True)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("krepair_new")

DEVNULL = subprocess.DEVNULL

def process_constraint_batch(batch, start_idx):
    """Process a batch of constraints and return string representations"""
    declarations = set()
    assertions = []

    for idx, (config_name, constraint_list) in enumerate(batch):
        try:
            for constraint in constraint_list:
                if isinstance(constraint, bytes):
                    constraint = constraint.decode('latin1')

                # Extract declarations
                for line in constraint.split('\n'):
                    if '(declare-fun' in line:
                        declarations.add(line.strip())

                # Extract assertions
                if '(assert' in constraint:
                    assert_start = constraint.find('(assert')
                    assert_end = constraint.rfind(')')
                    if assert_start != -1 and assert_end != -1:
                        assertion = constraint[assert_start:assert_end+1]
                        assertions.append(assertion)
        except Exception as e:
            print(f"Error processing constraint {start_idx + idx}: {e}")

    return (declarations, assertions)

def get_arch_formulas_dir(formulas: str, arch: str) -> str:
    """Helper to construct the formulas directory path for a given architecture."""
    assert arch is not None, "Arch name cannot be None"
    return os.path.join(formulas, f"{arch}_formulas.pkl")

class krepairDivQ:
    DECL_PATTERN = re.compile(r"(CONFIG_[A-Z0-9_]+)")  # compile once for efficiency

    def __init__(self, linux_ksrc: str, existing_config_path: str):
        self.complex_arch_constraints = []
        self.arch_smt2_str = ""
        self.dependency_graph = nx.DiGraph()
        self.unbootable_options = set()
        self.linux_ksrc = linux_ksrc
        self.patch_constraints = []
        self.patch_constraints_seen = set()
        self.patch_declarations = set()
        self.unit_constraints = defaultdict(list)
        self.arch_baseline_solver = None
        self.parsed_arch_constraints = None
        self.existing_config_path = existing_config_path
        # Initialize existing_config_constraints before building declarations:
        self.existing_config_constraints = []  # Stores constraints from .config
        self.merged_groups = {}
        self.build_all_declarations()

    def build_all_declarations(self):
        """
        Scan through all known constraints (patch_constraints and existing_config_constraints)
        and cache a set of SMT2 declarations for each encountered configuration symbol.
        """
        tokens = set()
        # Process constraints from existing config:
        for c in self.existing_config_constraints:
            # Use the sexpr representation for proper SMT2 syntax
            token_list = self.DECL_PATTERN.findall(c.sexpr())
            tokens.update(token_list)
        # Process constraints from patch constraints (they are strings)
        for c_str in self.patch_constraints:
            token_list = self.DECL_PATTERN.findall(c_str)
            tokens.update(token_list)
        # Build SMT2 declaration strings:
        self.patch_declarations = {f"(declare-const {token} Bool)" for token in tokens}
        print(f"Cached {len(self.patch_declarations)} declarations.")

    def get_arch_constraints(self, arch_name: str = "x86_64"):
        """Get arch constraints using existing functionality"""
        try:
            # Initialize Arch object
            self.arch = Arch(
                arch_name,
                linux_ksrc=self.linux_ksrc,
                arch_dir=get_arch_formulas_dir(self.linux_ksrc, arch_name),
                is_kclause_composite=True,
                kextract_version="next-20210426",
                loggerLevel=logging.INFO
            )

            kclause_path = os.path.join(get_arch_formulas_dir(self.linux_ksrc, arch_name), 'kclause')
            print(f"Looking for kclause at: {kclause_path}")

            if not os.path.exists(kclause_path) or os.path.getsize(kclause_path) == 0:
                print("Kclause file missing or empty, generating...")
                self.arch.generate_kclause()
            else:
                print(f"Found existing kclause file ({os.path.getsize(kclause_path)} bytes)")

            # Try to load the constraints directly using pickle
            try:
                with open(kclause_path, 'rb') as f:
                    self.compiled_arch_constraints = pickle.load(f)
            except Exception as e:
                print(f"Error loading kclause file directly: {e}")
                # Fallback to regenerating constraints
                self.compiled_arch_constraints = self.arch.load_kclause(
                    kclause_file=kclause_path,
                    is_composite=True
                )

            # Convert compiled_arch_constraints into Z3 expressions
            self.arch_constraints = []
            for constraint in self.compiled_arch_constraints:
                if isinstance(constraint, str):
                    # Convert strings to Z3 variables directly
                    z3_var = z3.Bool(constraint)
                    self.arch_constraints.append(z3_var)
                else:
                    self.arch_constraints.append(constraint)

            print(f"Successfully loaded {len(self.arch_constraints)} arch constraints")

        except Exception as e:
            print(f"Error in arch constraints processing: {e}")
            self.compiled_arch_constraints = []
            self.arch_constraints = []

    def get_complex_arch_constraints(self, arch_name: str = "x86_64", output_file: str = None):
        """
        this method builds a giant smt2 string and stores it in self.arch_smt2_str.
        instead of parsing to z3 objects, we keep it as text to avoid pickling errors.
        """
        try:
            # 1) load raw constraints (pickled) from 'kclause_path'
            kclause_path = os.path.join(self.linux_ksrc, f"{arch_name}_formulas.pkl", 'kclause')
            if not os.path.exists(kclause_path) or os.path.getsize(kclause_path) == 0:
                print("kclause file missing or empty, generating... (omitted here)")
                return

            with open(kclause_path, 'rb') as f:
                raw_constraints = pickle.load(f)
            print("loaded raw constraints")

            items = list(raw_constraints.items())
            cpu_count = mp.cpu_count()
            chunk_size = max(1, len(items) // (cpu_count * 2))
            batches = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

            print(f"processing {len(batches)} batches using {cpu_count} processes...")

            # 2) use mp.Pool to transform raw constraints => declarations/assertions
            with mp.Pool(processes=cpu_count) as pool:
                process_with_index = partial(process_constraint_batch)
                start_indices = range(0, len(items), chunk_size)
                batch_results = pool.starmap(process_with_index, zip(batches, start_indices))

            all_declarations = set()
            all_assertions = []
            for decls, asserts in batch_results:
                all_declarations.update(decls)
                all_assertions.extend(asserts)

            # 3) build the final big text
            combined_smt2 = "(set-logic QF_UF)\n"
            for decl in sorted(all_declarations):
                combined_smt2 += decl + "\n"
            for assertion in all_assertions:
                combined_smt2 += assertion + "\n"

            # note: do not parse here, just store as text
            self.arch_smt2_str = combined_smt2
            print(f"[debug] built arch_smt2_str with length: {len(self.arch_smt2_str)} chars")

            try:
                # First, parse it normally
                self.parsed_arch_constraints = z3.parse_smt2_string(self.arch_smt2_str)
                print(f"[DEBUG] Parsed {len(self.parsed_arch_constraints)} arch constraints")

                print("\n[DEBUG] First 10 parsed arch constraints:")
                for constraint in self.parsed_arch_constraints[:10]:
                    print(f"  - {constraint} (Type: {type(constraint)})")

                # Now, manually reconstruct logical constraints like the old implementation
                reconstructed_constraints = []
                for constraint in self.parsed_arch_constraints:
                    if isinstance(constraint, z3.BoolRef):
                        reconstructed_constraints.append(constraint)
                    else:
                        # Convert to a Z3 boolean expression if it's in string form
                        try:
                            reconstructed_constraints.append(eval(str(constraint)))
                        except Exception as e:
                            print(f"[WARNING] Failed to convert constraint: {constraint}, error: {e}")

                self.parsed_arch_constraints = reconstructed_constraints
                print(f"[DEBUG] Reconstructed {len(self.parsed_arch_constraints)} Z3 constraints")
            except Exception as e:
                print(f"[ERROR] Failed to parse arch constraints: {e}")
                self.parsed_arch_constraints = []

            self.init_arch_baseline_solver()

            # optionally write to file
            if output_file:
                with open(output_file, "w", encoding='utf-8') as out_f:
                    out_f.write(combined_smt2)
                print(f"complex constraints written to {output_file}")

        except Exception as e:
            print(f"error loading complex constraints: {e}")
            self.arch_smt2_str = ""

    def init_arch_baseline_solver(self):
        """
        Initialize a baseline solver that contains the full arch constraints,
        including extra architecture-specific assertions for x86_64.
        This is done once so we avoid repeatedly parsing the arch constraints.
        """
        try:
            # Create a new context and solver for the arch constraints.
            self.arch_ctx = z3.Context()
            self.arch_baseline_solver = z3.Solver(ctx=self.arch_ctx)

            if self.arch_smt2_str and self.arch_smt2_str.strip():
                # Parse the combined arch constraints from arch_smt2_str.
                arch_exprs = list(z3.parse_smt2_string(self.arch_smt2_str, ctx=self.arch_ctx))

                # If the architecture is x86_64, add extra architecture-specific assertions.
                if self.arch.name == "x86_64":
                    extra_decls = []
                    extra_assertions = []
                    # Define extra assertions as pairs (constant, assertion).
                    # Note: We rename BITS=64 to BITS_64 and BITS=32 to BITS_32 to avoid syntax errors.
                    extra_pairs = [
                        ("CONFIG_X86", "(assert CONFIG_X86)"),
                        ("CONFIG_X86_64", "(assert CONFIG_X86_64)"),
                        ("CONFIG_X86_32", "(assert (not CONFIG_X86_32))"),
                        ("BITS==64", "(assert BITS==64)"),
                        ("BITS==32", "(assert (not BITS==32))")
                    ]
                    for const, assertion in extra_pairs:
                        extra_decls.append(f"(declare-const {const} Bool)")
                        extra_assertions.append(assertion)

                    # Negative assertions for all other architectures.
                    other_archs = [
                        "ALPHA", "ARC", "ARM", "ARM64", "C6X", "CSKY", "H8300", "HEXAGON", "IA64",
                        "LOONGARCH", "M68K", "MICROBLAZE", "MIPS", "NDS32", "NIOS2", "OPENRISC",
                        "PARISC", "PPC", "PPC32", "PPC64", "RISCV", "S390", "SPARC", "SPARC32",
                        "SPARC64", "SUPERH", "SUPERH32", "SUPERH64", "UML", "UNICORE32", "XTENSA"
                    ]
                    for arch in other_archs:
                        const = f"CONFIG_{arch}"
                        extra_decls.append(f"(declare-const {const} Bool)")
                        extra_assertions.append(f"(assert (not {const}))")
                    # Add the extra assertion for CONFIG_BROKEN.
                    extra_decls.append("(declare-const CONFIG_BROKEN Bool)")
                    extra_assertions.append("(assert (not CONFIG_BROKEN))")

                    # Build a string that combines the declarations and assertions.
                    extra_arch_str = "(set-logic QF_UF)\n" + "\n".join(extra_decls + extra_assertions)
                    try:
                        extra_arch_exprs = list(z3.parse_smt2_string(extra_arch_str, ctx=self.arch_ctx))
                        arch_exprs.extend(extra_arch_exprs)
                    except Exception as e:
                        print(f"[ERROR] Failed to parse extra arch constraints: {e}")
                        raise

                # Add all the architecture constraints to the baseline solver.
                self.arch_baseline_solver.add(arch_exprs)
                print(f"[debug] arch baseline solver initialized with {len(arch_exprs)} constraints")
            else:
                print("[debug] arch_smt2_str is empty; arch baseline solver not initialized")
        except Exception as e:
            print(f"error initializing arch baseline solver: {e}")
            self.arch_baseline_solver = None

    def build_kconfig_dependency_graph(self, content):
        """
        Build dependency graph from kextract output using kclause-style parsing.
        We'll store edges in the direction: DEPENDENCY -> DEPENDENT.
        So 'dep' lines cause edges: each item in expr -> var
        """
        G = nx.DiGraph()
        dep_exprs = {}
        rev_dep_exprs = {}
        selects = {}
        config_types = {}

        print(f"[Debug] Starting to build dependency graph")

        def parse_kconfig_line(line):
            if line.strip():
                try:
                    instr, data = line.strip().split(" ", 1)
                    return instr, data
                except ValueError:
                    return None, None
            return None, None

        def parse_dependency_expr(expr):
            configs = set()
            expr = expr.strip('()')
            for term in expr.split(' and '):
                term = term.strip()
                if ' or ' in term:
                    or_terms = term.strip('()').split(' or ')
                    for or_term in or_terms:
                        if or_term.startswith('CONFIG_'):
                            configs.add(or_term)
                elif term.startswith('not '):
                    term = term.replace('not ', '').strip('()')
                    if term.startswith('CONFIG_'):
                        configs.add(term)
                elif term.startswith('CONFIG_'):
                    configs.add(term)
            return list(configs)

        print("Parsing kconfig data...")
        for line in content.split('\n'):
            instr, data = parse_kconfig_line(line)
            if not instr:
                continue

            if instr == "config":
                var, type_name = data.split(" ", 1)
                config_types[var] = type_name
                G.add_node(var, type=type_name)

            elif instr == "dep":
                # e.g. 'dep CONFIG_DE2104X (CONFIG_NETDEVICES and ... )'
                try:
                    var, expr = data.split(" ", 1)
                    dep_exprs[var] = expr
                    deps = parse_dependency_expr(expr)
                    # We store: each 'dep' is a prerequisite for 'var', so dep -> var
                    for dep in deps:
                        G.add_node(var)
                        G.add_node(dep)
                        G.add_edge(dep, var, type='depends')
                except Exception as e:
                    print(f"Error processing dep line: {data}")
                    print(f"Error: {e}")

            elif instr == "select":
                # e.g. 'select CONFIG_X CONFIG_Y (expr)'
                try:
                    selected_var, selecting_var, expr = data.split(" ", 2)
                    if selected_var not in selects:
                        selects[selected_var] = {}
                    if selecting_var not in selects[selected_var]:
                        selects[selected_var][selecting_var] = set()
                    selects[selected_var][selecting_var].add(expr)

                    G.add_node(selecting_var)
                    G.add_node(selected_var)
                    # For 'select', interpret "selecting_var -> selected_var" as
                    # "selected_var is a prerequisite to selecting_var" or the reverse.
                    # Typically we do "selected_var -> selecting_var" if 'select' means:
                    #   "If selecting_var is on, it forcibly sets selected_var."
                    # Then selected_var is effectively a 'dependency' if it can't be turned on.
                    #
                    # But many treat "select" as "selecting_var depends on selected_var".
                    # For consistency with 'dep', do 'selected_var -> selecting_var' (i.e. "if X is not possible, Y cannot select it").
                    # However, if you prefer the opposite, be consistent throughout.
                    # Let's do the same direction as 'dep': dependency -> dependent.
                    #
                    # So: "CONFIG_X is forced on by CONFIG_Y" => "X is a needed item, Y can't be valid if X can't be turned on"
                    # so X -> Y
                    G.add_edge(selecting_var, selected_var, type='selects')
                except Exception as e:
                    print(f"Error processing select line: {data}")
                    print(f"Error: {e}")

            elif instr == "rev_dep":
                # e.g. "rev_dep CONFIG_ISA_BUS_API (CONFIG_GPIO_104_DIO_48E and ... )"
                try:
                    var, expr = data.split(" ", 1)
                    rev_dep_exprs[var] = expr
                    deps = parse_dependency_expr(expr)
                    # "rev_dep" is typically a reversed approach: "these 'deps' forcibly rely on var."
                    # If we want the same direction "dependency -> dependent",
                    # then the 'dependency' is 'var', the 'dependent' is each item in 'deps'.
                    # So we do: var -> dep
                    for dep in deps:
                        G.add_node(var)
                        G.add_node(dep)
                        G.add_edge(var, dep, type='reverse_depends')
                except Exception as e:
                    print(f"Error processing rev_dep line: {data}")
                    print(f"Error: {e}")

        print(f"[Debug] Built dependency graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Print a small number of 'select' edges
        select_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'selects']
        print(f"Total 'select' edges found: {len(select_edges)}")
        print("Printing up to 10 'select' edges:\n")
        for i, (u, v) in enumerate(select_edges[:10]):
            print(f"{i + 1}. {u} -> {v} (type: 'selects')")

        return G

    def analyze_kconfig_graph(self, G):
        """Analyze the Kconfig dependency graph"""
        print("\nDependency Graph Analysis:")
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {G.number_of_edges()}")

        edge_types = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print("\nEdge types:")
        for edge_type, count in edge_types.items():
            print(f"  {edge_type}: {count}")

        in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
        print("\nMost depended-upon configs:")
        for node, degree in in_degrees[:10]:
            print(f"  {node}: {degree} incoming edges")

        out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
        print("\nConfigs with most dependencies:")
        for node, degree in out_degrees[:10]:
            print(f"  {node}: {degree} outgoing edges")

    def process_kconfig_dependencies(self, kextract_content):
        """Process Kconfig dependencies and generate analysis"""
        print("Building dependency graph...")
        self.dependency_graph = self.build_kconfig_dependency_graph(kextract_content)  # Store in self
        print(f"[Debug] Stored dependency graph with {self.dependency_graph.number_of_nodes()} nodes")

        print("\nAnalyzing graph...")
        self.analyze_kconfig_graph(self.dependency_graph)  # Use stored graph

        return self.dependency_graph  # Can still return if needed

    def is_arch_specific_unit(self, unit: str, target_arch: str, accept_x86: bool = True) -> bool:
        """
        Check if a unit is architecture-specific and doesn't match target architecture
        Returns True if unit should be skipped (not compatible with target_arch)

        Parameters:
            unit: The compilation unit path
            target_arch: The target architecture (e.g., "x86_64")
            accept_x86: Whether to also accept "x86" as compatible with "x86_64"
        """
        if unit.startswith("arch/"):
            # Get architecture from subdirectory
            unit_arch = unit.split('/')[1]  # Gets the arch name from arch/NAME/...

            # Accept both target_arch and "x86" when target is "x86_64"
            if accept_x86 and target_arch == "x86_64" and unit_arch == "x86":
                return False  # Don't skip x86 units when targeting x86_64

            return unit_arch != target_arch  # Skip if unit architecture doesn't match target
        return False  # Not architecture-specific, so don't skip

    def parse_patch_configs_file(self, file_path: str, target_arch: str = "x86_64"):
        """
        Handles lines like:
          - Single config:  CONFIG_FOO
          - Single-line s-expr: (and CONFIG_FOO CONFIG_BAR)
          - Multi-line s-expr:
            (or CONFIG_FOO
                (and CONFIG_FOO CONFIG_BAR)
                (and CONFIG_FOO CONFIG_BAZ))
          - # Unit: lines to switch units
        Skips lines that contain leftover "(declare-const", "(assert", or "(check-sat)"
        to avoid re-parsing old solver lines.
        """
        def strip_defined_expressions(line: str) -> str:
            """
            Replaces (defined X) or |(defined X)| with just X.
            """
            pattern = r'\(\s*defined\s+([^\s)]+)\)|\|\(\s*defined\s+([^\s)]+)\)\|'
            def replacer(m):
                return m.group(1) if m.group(1) else m.group(2)
            return re.sub(pattern, replacer, line)

        def is_parentheses_balanced(s: str) -> bool:
            """
            Checks if parentheses are balanced overall in s.
            """
            count = 0
            for ch in s:
                if ch == '(':
                    count += 1
                elif ch == ')':
                    count -= 1
                if count < 0:
                    return False
            return (count == 0)

        def balance_parentheses(s: str) -> str:
            """
            Fix minor mismatches by removing trailing ')' or adding missing ')'.
            """
            s = s.strip()
            open_count = s.count('(')
            close_count = s.count(')')
            while close_count > open_count and s.endswith(')'):
                s = s[:-1]
                close_count -= 1
            while open_count > close_count:
                s += ')'
                close_count += 1
            return s

        def collect_tokens_for_decl(line: str):
            pattern = r"\b(CONFIG_[A-Z0-9_]+)\b"
            found_tokens = re.findall(pattern, line)
            return list(set(found_tokens))  # Remove duplicates

        def parse_and_store_expression(expr_str: str, current_unit: str):
            print("[DEBUG] parse_and_store_expression input =", repr(expr_str))
            expr_str = expr_str.strip()
            if not expr_str:
                return

            expr_str = strip_defined_expressions(expr_str)  # your existing helper
            expr_str = balance_parentheses(expr_str)        # your existing helper

            # Convert operators into proper SMT-LIB syntax: add a leading '(' and a trailing space.
            expr_str = re.sub(r'\bAnd\(', '(and ', expr_str)
            expr_str = re.sub(r'\bOr\(', '(or ', expr_str)
            # Remove commas (SMT-LIB uses whitespace separation)
            expr_str = expr_str.replace(',', '')

            # Gather config tokens for local declare
            configs = collect_tokens_for_decl(expr_str)
            configs = list(set(configs))  # de-dupe
            declare_lines = [f"(declare-const {c} Bool)" for c in sorted(configs)]
            for dl in declare_lines:
                self.patch_declarations.add(dl)

            # 1st attempt: parse as is
            parse_ok = try_parse_smt_expr(expr_str, declare_lines)
            if not parse_ok:
                print(f"[DEBUG] parse failed, attempting naive fix. Original: {expr_str}")
                fixed_expr = try_fix_expression(expr_str)
                if fixed_expr != expr_str:
                    print(f"[DEBUG] Applied hack to fix malformed constraint:")
                    print(f"  Original: {expr_str}")
                    print(f"  Fixed:    {fixed_expr}")
                parse_ok = try_parse_smt_expr(fixed_expr, declare_lines)
                if parse_ok:
                    print(f"[DEBUG] parse succeeded after fix => {fixed_expr}")
                    expr_str = fixed_expr
                else:
                    print(f"[DEBUG] parse_and_store_expression => STILL FAIL, skipping expression.")
                    return

            assertion_str = f"(assert {expr_str})"
            if assertion_str not in self.patch_constraints_seen:
                self.patch_constraints.append(assertion_str)
                self.unit_constraints[current_unit].append(assertion_str)
                self.patch_constraints_seen.add(assertion_str)
                for c in configs:
                    self.unit_configs[current_unit].add(c)

        def try_fix_expression(expr_str: str) -> str:
            """
            Very naive 'fix':
              1. If there's a trailing '(and CONFIG_X ...' or '(or CONFIG_X ...' with no closing ')', add them.
              2. If open parentheses > close parentheses, add that many closing ')'.
              3. If there's trailing 'and', 'or' with no actual second argument, remove them.
            """
            expr_str = expr_str.strip()

            # (A) Count parentheses
            open_count = expr_str.count('(')
            close_count = expr_str.count(')')
            if close_count < open_count:
                print(f"[DEBUG] Fixing unmatched parentheses: Adding {open_count - close_count} closing ')' to balance")
            # If we have more '(' than ')', tack on the difference:
            while close_count < open_count:
                expr_str += ')'
                close_count += 1

            # (B) If there's a trailing operator 'and' or 'or' with no tokens after it, remove it:
            trailing_op_pattern = r'(.*)(\(and [^)]+|\(or [^)]+)$'
            match = re.match(trailing_op_pattern, expr_str)
            if match:
                print(f"[DEBUG] Removing incomplete trailing operator in expression: {expr_str}")
                expr_str = match.group(1).strip()
                # Now we might need to rebalance parentheses again
                expr_str = balance_parentheses(expr_str)

            return expr_str

        def try_parse_smt_expr(expr_str: str, declarations: List[str]) -> bool:
            """
            Attempt to parse a single expression string along with the given declare-const lines.
            Return True if parse is successful, else False.
            """
            ctx = z3.Context()
            solver = z3.Solver(ctx=ctx)
            snippet_for_one_assert = "\n".join([
                *declarations,
                f"(assert {expr_str})",
                "(check-sat)"
            ])
            try:
                parsed = z3.parse_smt2_string(snippet_for_one_assert, ctx=ctx)
                solver.add(parsed[:-1])  # everything except the (check-sat)
                result = solver.check()
                return (result == z3.sat or result == z3.unknown)
            except Exception as e:
                print(f"[DEBUG] Failed to parse SMT expression: {e}")
                return False

        ##########################
        # Main parse logic
        ##########################
        current_unit = None
        block_lines = []

        # Clear out previous parse results
        self.unit_constraints = defaultdict(list)
        self.unit_configs = defaultdict(set)
        self.config_vars_cache = {}
        # self.patch_constraints, self.patch_declarations presumably are in __init__ or so

        def flush_block():
            nonlocal block_lines, current_unit
            if not block_lines:
                return

            expr_buffer = []
            for line in block_lines:
                stripped = line.strip()
                if not stripped:
                    continue

                print("[DEBUG] Adding to expr_buffer:", repr(stripped))
                expr_buffer.append(stripped)
                joined_expr = " ".join(expr_buffer)
                print("[DEBUG] Current accumulated expression:", repr(joined_expr))
                if is_parentheses_balanced(joined_expr):
                    print("[DEBUG] Expression is balanced, flushing:", repr(joined_expr))
                    parse_and_store_expression(joined_expr, current_unit)
                    expr_buffer = []  # reset for the next expression

            # Flush any remaining expression that might not have been balanced (optional)
            if expr_buffer:
                joined_expr = " ".join(expr_buffer).strip()
                print("[DEBUG] Flushing leftover expression:", repr(joined_expr))
                parse_and_store_expression(joined_expr, current_unit)
            block_lines = []

        # Read the file line-by-line
        with open(file_path, "r") as f:
            for raw_line in f:
                line = raw_line.rstrip()

                # 1) Handle "# Unit:" lines
                if line.startswith("# Unit:"):
                    # Flush the accumulated block for the previous unit
                    flush_block()
                    candidate_unit = line[len("# Unit:"):].strip()
                    target_arch = "x86_64"  # The target architecture we're working with
                    if self.is_arch_specific_unit(candidate_unit, target_arch, accept_x86=True):
                        print(f"[DEBUG] Skipping unit: {candidate_unit} for architecture {target_arch}")
                        current_unit = None
                    else:
                        current_unit = candidate_unit
                        print(f"[DEBUG] Processing unit: {current_unit}")
                    continue

                # 2) Ignore comments (except for "# Unit:" handled above)
                if line.startswith("#"):
                    continue

                # 3) Skip lines that contain leftover solver output.
                if "(declare-const" in line or "(assert" in line or "(check-sat" in line:
                    continue

                # 4) Instead of flushing on blank lines, just accumulate every non-None line
                if current_unit is None:
                    continue

                block_lines.append(line)

            # End of file => flush remaining block
            flush_block()

        # Summaries
        total_exprs = sum(len(v) for v in self.unit_constraints.values())
        print("\nFinal Summary from parse_patch_configs_file:")
        print(f"  Units found: {len(self.unit_constraints)}")
        print(f"  Total expressions: {total_exprs}")
        print(f"  Total patch configs: {len(self.patch_constraints)}\n")

        print("\n[Debug] All Patch Constraints (in read order):")
        for c in self.patch_constraints:
            print(c)
        print(f"\n[Debug] Total Patch Constraints: {len(self.patch_constraints)}")

    def remove_unbootable_options(self, unbootable_file_path: str):
        """
        Remove unbootable configs from patch data structures.
        BUT only remove those options that *select* these unbootable configs
        (i.e. traverse only 'selects' edges). This way, an unbootable config
        won't get re-enabled by anything that tries to select it.
        """

        # 1) Read unbootable from file
        unbootable_options = set()
        with open(unbootable_file_path, 'r') as f:
            for line in f:
                cfg = line.strip()
                if cfg:
                    unbootable_options.add(cfg)

        to_remove = set()

        # 2) For each unbootable config, do BFS over 'selects' edges
        for ub_cfg in unbootable_options:
            to_remove.add(ub_cfg)
            if ub_cfg not in self.dependency_graph.nodes:
                continue

            queue = [ub_cfg]
            visited = {ub_cfg}

            while queue:
                current = queue.pop()
                for succ in self.dependency_graph.successors(current):
                    edge_data = self.dependency_graph.get_edge_data(current, succ)
                    e_type = edge_data.get('type', None)
                    # ONLY follow 'selects' edges here
                    if e_type == 'selects' and succ not in visited:
                        visited.add(succ)
                        queue.append(succ)
                        to_remove.add(succ)

        print("\n[Debug] Removing the following unbootable options + their selectors:")
        for cfg in sorted(to_remove):
            print(f"  {cfg}")

        # 3) Filter out references to these options in self.patch_constraints
        new_patch_constraints = []
        for c_expr in self.patch_constraints:
            c_str = str(c_expr)
            # If expression does NOT contain any unbootable config, keep it
            if not any(r in c_str for r in to_remove):
                new_patch_constraints.append(c_expr)

        self.patch_constraints = new_patch_constraints
        # Rebuild the seen set so it stays in sync
        self.patch_constraints_seen = set(new_patch_constraints)

        # 4) Filter from each unit in unit_constraints and unit_configs
        for unit in list(self.unit_constraints.keys()):
            old_constraints = self.unit_constraints[unit]
            new_constraints = []
            for c_expr in old_constraints:
                if not any(r in str(c_expr) for r in to_remove):
                    new_constraints.append(c_expr)
            self.unit_constraints[unit] = new_constraints

            old_unit_configs = self.unit_configs[unit]
            new_unit_configs = {cfg for cfg in old_unit_configs if cfg not in to_remove}
            self.unit_configs[unit] = new_unit_configs

        print("\n[Debug] Finished removing unbootable options (via 'selects' edges only).")
        print(f"  Unbootable configs from file: {unbootable_options}")
        print(f"  Total removed (including 'selectors'): {len(to_remove)}")
        print("  Updated self.patch_constraints, self.unit_constraints, self.unit_configs accordingly.")

    # function to basically take patch_constraints and start iterative solve (add sat until unsat).
    def check_constraints_until_unsat_parallel(self, num_threads=24):
        """
        Main function that performs parallel checks on patch constraints.
        This version groups patch constraints by their compilation unit using
        self.unit_constraints and then partitions the sorted groups into a target
        number of chunks.

        Assumes:
          - self.patch_constraints is a list of constraint strings.
          - self.unit_constraints is a defaultdict(list) mapping unit names to lists
            of constraint strings.
          - self.arch_smt2_str and self.patch_declarations are already set.
        """
        from collections import defaultdict
        # --- Group patch constraints by compilation unit ---
        # Assume self.unit_constraints is already populated.
        groups_by_unit = self.unit_constraints  # e.g., {"fs/smb/client/connect.o": [...], ...}

        # Sort the unit keys
        unit_keys = sorted(groups_by_unit.keys())

        # --- Partition unit groups into blocks if there are too many groups ---
        if len(self.patch_constraints) < 1000:
            TARGET_NUM_CHUNKS = 50  # Aim for 50 chunks if under 1k constraints
        else:
            TARGET_NUM_CHUNKS = max(1, len(self.patch_constraints) // 6)  # Otherwise, divide by 8

        if len(unit_keys) > TARGET_NUM_CHUNKS:
            block_size = (len(unit_keys) + TARGET_NUM_CHUNKS - 1) // TARGET_NUM_CHUNKS
            print(f"Partitioning {len(unit_keys)} unit groups into blocks of about {block_size} keys each.")
            new_groups = {}
            for i in range(0, len(unit_keys), block_size):
                merged_key = "_".join(unit_keys[i:i+block_size])
                merged_constraints = []
                for key in unit_keys[i:i+block_size]:
                    merged_constraints.extend(groups_by_unit[key])
                new_groups[merged_key] = merged_constraints
            merged_groups = new_groups
        else:
            merged_groups = groups_by_unit

        # Now sort the merged group keys.
        final_keys = sorted(merged_groups.keys())

        # Build chunks from these groups.
        chunks = []
        starts = []
        cumulative = 0
        for key in final_keys:
            chunk = merged_groups[key]
            chunks.append(chunk)
            starts.append(cumulative)
            cumulative += len(chunk)

        total_constraints = len(self.patch_constraints)
        total_chunks = len(chunks)
        print(f"Grouped into {total_chunks} chunks (total {total_constraints} constraints) after partitioning by unit.")

        # Build a unified header from patch declarations.
        all_declarations = "\n".join(sorted(self.patch_declarations))
        smt_template = f"""(set-logic QF_UF)
    {all_declarations}
    %CONSTRAINTS%
    (check-sat)
    """
        # Build full SMT scripts for each chunk.
        chunk_scripts = []
        for i, chunk_constraints in enumerate(chunks):
            chunk_text = "\n".join(chunk_constraints)
            full_script = smt_template.replace("%CONSTRAINTS%", chunk_text)
            chunk_scripts.append(full_script)

        # --- 3) Parallel processing using ProcessPoolExecutor ---
        manager = mp.Manager()
        shared_data = manager.Namespace()
        shared_data.counter = 0
        pbar = tqdm(total=total_constraints, desc="Processing constraints")

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {
                executor.submit(
                    process_complete_smt_script,
                    chunk_scripts[i],          # Chunk's SMT text.
                    i,                         # Chunk index.
                    starts[i],                 # Starting index.
                    chunks[i],                 # Local constraints for this chunk.
                    self.patch_constraints,    # Full list (if needed).
                    self.arch_smt2_str,        # Raw arch constraints text.
                    shared_data,               # Shared counter.
                    self.patch_declarations,   # Patch declarations.
                    self.arch.name             # Architecture name.
                ): i for i in range(total_chunks)
            }

            results_by_index = [None] * total_chunks
            all_temp_unsat = {}

            while any(not fut.done() for fut in future_to_index):
                current = shared_data.counter
                pbar.n = current
                pbar.refresh()
                time.sleep(0.5)
            pbar.n = shared_data.counter
            pbar.refresh()
            pbar.close()

            for fut in as_completed(future_to_index):
                i = future_to_index[fut]
                try:
                    valid_idx, temp_unsat, chunk_id = fut.result()
                    results_by_index[i] = valid_idx
                    if temp_unsat:
                        all_temp_unsat[chunk_id] = temp_unsat
                except Exception as e:
                    print(f"Error in chunk {i}: {e}")

        # --- 4) Post-processing: Organize, update, merge ---
        all_valid_indexes = []
        for i, chunk_result in enumerate(results_by_index):
            all_valid_indexes.extend(chunk_result or [])
            print(f"Chunk {i+1}/{total_chunks}: {len(chunk_result or [])} valid constraints")

        orig_patch_constraints = self.patch_constraints.copy()
        self._update_patch_constraints(all_valid_indexes)
        print(f"Successfully processed {len(self.patch_constraints)} constraints")
        print(f"Found {len(all_temp_unsat)} unsat-causing constraints")
        self._print_temp_unsat(all_temp_unsat)

        valid_by_chunk = self._organize_valid_constraints(all_valid_indexes, starts, orig_patch_constraints)
        never_sat = self._process_temp_unsat(all_temp_unsat, valid_by_chunk)
        all_temp_unsat.clear()
        self._finalize_results(valid_by_chunk, never_sat, all_temp_unsat)

        valid_by_chunk, completed_sets = self._attempt_merge_chunks(valid_by_chunk)
        self.patch_constraints = []
        for chunk_id in sorted(valid_by_chunk.keys()):
            self.patch_constraints.extend(valid_by_chunk[chunk_id])

        print(f"Final constraints count after merging: {len(self.patch_constraints)}")
        print(f"Completed sets (never merged): {len(completed_sets)}")

        print("\n--- Final Grouped Constraints ---")
        for chunk_id, constraints in sorted(valid_by_chunk.items()):
            print(f"\nChunk {chunk_id}: {len(constraints)} constraints")
            for constraint in constraints[:5]:
                print(f"  - {constraint.strip()}")
            chunk_filename = f"final_chunk_{chunk_id}.smt2"
            with open(chunk_filename, "w") as chunk_file:
                chunk_file.write("\n".join(constraints))

        print("\n--- Completed Sets (Unmerged) ---")
        for chunk_id in sorted(completed_sets):
            print(f"Chunk {chunk_id} was never merged.")

        self.merged_groups = valid_by_chunk

        return {
            "added_constraints": len(self.patch_constraints),
            "temp_unsat": all_temp_unsat,
            "never_sat": never_sat
        }

    def _update_patch_constraints(self, all_valid_indexes):
        max_valid_index = len(self.patch_constraints) - 1
        filtered = [i for i in all_valid_indexes if i <= max_valid_index]
        self.patch_constraints = [self.patch_constraints[i] for i in filtered]

    def _print_temp_unsat(self, all_temp_unsat):
        """Print temporary UNSAT constraints"""
        print("\nTempUnsat constraints by chunk:")
        for chunk_id, constraints in sorted(all_temp_unsat.items()):
            print(f"\nChunk {chunk_id}:")
            for constraint in constraints:
                print(f"  - {constraint.strip()}")

    def _organize_valid_constraints(self, all_valid_indexes, starts, orig_constraints):
        """Organize valid constraints into chunks using the original constraints list.
           orig_constraints should be the patch constraints list before it is updated.
        """
        valid_by_chunk = defaultdict(list)
        for idx in all_valid_indexes:
            # determine which chunk this index belongs to
            chunk_id = next(
                (i+1 for i, start in enumerate(starts)
                 if start <= idx < (starts[i+1] if i < len(starts)-1 else len(orig_constraints))),
                None
            )
            if chunk_id is not None:
                valid_by_chunk[chunk_id].append(orig_constraints[idx])
        return valid_by_chunk

    def _process_temp_unsat(self, all_temp_unsat, valid_by_chunk):
        """
        For each unsat constraint in all_temp_unsat, try to add it to an existing chunk.
        If a constraint remains unsatisfiable after trying 5 chunks, it is added to never_sat.
        """
        never_sat = set()
        completed_sets = set()

        def get_declarations_from_constraints(constraints):
            tokens = set()
            for cons in constraints:
                tokens.update(re.findall(r"(CONFIG_[A-Z0-9_]+)", cons))
            decl_lines = [f"(declare-const {t} Bool)" for t in sorted(tokens)]
            return "\n".join(decl_lines)

        def test_constraint(constraint, target_chunk_constraints):
            try:
                # Use the baseline arch solver’s context.
                ctx = self.arch_baseline_solver.ctx
                solver = z3.Solver(ctx=ctx)
                # Add the baseline arch constraints.
                solver.append(*self.arch_baseline_solver.assertions())

                # Extract all CONFIG tokens from both the target constraints and the candidate.
                import re
                def extract_tokens(constraints):
                    tokens = set()
                    for cons in constraints:
                        tokens.update(re.findall(r"(CONFIG_[A-Z0-9_]+)", cons))
                    return tokens

                group_tokens = extract_tokens(target_chunk_constraints)
                candidate_tokens = extract_tokens([constraint])
                all_tokens = sorted(group_tokens.union(candidate_tokens))

                # Build a declarations block.
                decl_block = "\n".join(f"(declare-const {tok} Bool)" for tok in all_tokens)

                # Now build a “clean” SMT2 script.
                script = "(set-logic QF_UF)\n" + decl_block + "\n" \
                         + "\n".join(target_chunk_constraints) + "\n" \
                         + constraint

                # Debugging output
                print("Generated SMT2 script for testing constraint:\n", script)

                # Parse and add to the solver.
                parsed = z3.parse_smt2_string(script, ctx=ctx)
                solver.add(parsed)

                return solver.check() == z3.sat
            except Exception as e:
                print(f"  ! error: {str(e)[:100]}")
                return False

        for source_chunk_id, unsat_constraints in all_temp_unsat.items():
            print(f"\nTrying constraints from chunk {source_chunk_id}:")
            for constraint in unsat_constraints:
                print(f"\nTesting: {constraint.strip()}")
                best_chunk = None
                attempts = 0

                # Sort chunks by size (excluding the source chunk)
                sorted_chunks = sorted(
                    [(cid, cons) for cid, cons in valid_by_chunk.items() if cid != source_chunk_id],
                    key=lambda x: len(x[1])
                )

                # Try the chunks
                for target_chunk_id, chunk_constraints in sorted_chunks:
                    if attempts >= 3:
                        print(f"  × Stopping after 3 attempts, adding to never_sat: {constraint.strip()}")
                        never_sat.add(constraint)
                        break

                    if test_constraint(constraint, chunk_constraints):
                        best_chunk = target_chunk_id
                        print(f"  ✓ Satisfiable with chunk {target_chunk_id} (size: {len(chunk_constraints)})")
                        break
                    else:
                        print(f"  × Not satisfiable with chunk {target_chunk_id}")
                    attempts += 1

                # Handle the outcome
                if best_chunk is not None:
                    valid_by_chunk[best_chunk].append(constraint)
                    print(f"  → Added to chunk {best_chunk} (new size: {len(valid_by_chunk[best_chunk])})")
                else:
                    # Add to never_sat if:
                    # 1. We tried all available chunks and none worked, or
                    # 2. We hit the 3-attempt limit
                    if attempts >= min(3, len(sorted_chunks)):
                        print(f"  → Adding to never_sat after trying {attempts} chunks: {constraint.strip()}")
                        never_sat.add(constraint)

        return never_sat

    def _finalize_results(self, valid_by_chunk, never_sat, all_temp_unsat):
        """Finalize and print results"""
        # Update patch constraints from organized chunks
        self.patch_constraints = []
        for chunk_id in sorted(valid_by_chunk.keys()):
            self.patch_constraints.extend(valid_by_chunk[chunk_id])

        print("\nFinal Results:")
        print(f"Total valid constraints: {len(self.patch_constraints)}")
        print(f"Total TempUnsat constraints: {sum(len(c) for c in all_temp_unsat.values())}")
        print(f"NeverSat constraints: {len(never_sat)}")

        if never_sat:
            print("\nNeverSat constraints:")
            for constraint in never_sat:
                print(f"  - {constraint.strip()}")

    def _attempt_merge_chunks(self, valid_by_chunk):
        """
        Perform balanced merging of groups of patch constraints by pairing groups
        with similar sizes.
        """
        tried_chunks = defaultdict(set)
        completed_sets = set()
        iteration = 0

        while True:
            # Order chunks by the number of constraints (smallest first)
            chunk_ids = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
            if not chunk_ids:
                break

            iteration += 1
            if iteration > 100:
                print("Breaking merge loop after 100 iterations to prevent infinite loop.")
                break

            print(f"\n=== Iteration {iteration}: Balanced Merge by Similar Size ===")
            print(f"Remaining chunks: {chunk_ids}")

            iteration_merged = False
            used_this_pass = set()

            # Process each chunk in order; try to merge with the closest-in-size partner.
            for i, c1 in enumerate(chunk_ids):
                if c1 in completed_sets or c1 in used_this_pass:
                    continue

                best_candidate = None
                best_diff = float('inf')
                # Look for a partner among the chunks after c1.
                for j in range(i + 1, len(chunk_ids)):
                    c2 = chunk_ids[j]
                    if c2 in completed_sets or c2 in used_this_pass or c2 in tried_chunks[c1]:
                        continue
                    # Calculate the size difference.
                    size_diff = abs(len(valid_by_chunk[c1]) - len(valid_by_chunk[c2]))
                    if size_diff < best_diff:
                        best_diff = size_diff
                        best_candidate = c2

                if best_candidate is not None:
                    c2 = best_candidate
                    print(f"\nPass: Trying to merge Chunk {c1} ({len(valid_by_chunk[c1])} constraints) "
                          f"with Chunk {c2} ({len(valid_by_chunk[c2])} constraints)...")
                    merged_constraints = valid_by_chunk[c1] + valid_by_chunk[c2]
                    if self._test_chunk_satisfiability(merged_constraints):
                        print(f"  ✓ Merge success: {c1} + {c2} → {len(merged_constraints)} constraints")
                        # Update group c1 with the merged constraints.
                        valid_by_chunk[c1] = merged_constraints
                        # Remove c2 from the dictionary.
                        del valid_by_chunk[c2]
                        used_this_pass.add(c1)
                        used_this_pass.add(c2)
                        tried_chunks[c1].add(c2)
                        tried_chunks[c2].add(c1)
                        iteration_merged = True
                    else:
                        print(f"  ✗ Merge failed: {c1} + {c2} → UNSAT")
                        tried_chunks[c1].add(c2)
                        tried_chunks[c2].add(c1)

            if not iteration_merged:
                print("No merges succeeded in this iteration. Stopping.")
                break
            else:
                print("Merges happened; starting a new iteration...")

        print("\nFinal Merging Completed.")
        final_chunks = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
        print(f"Remaining Chunks: {final_chunks}")
        print(f"Completed Sets: {len(completed_sets)}")
        return valid_by_chunk, completed_sets

    def _test_chunk_satisfiability(self, constraints):
        # Ensure the baseline solver is initialized.
        if self.arch_baseline_solver is None:
            print("[WARNING] arch_baseline_solver is None, reinitializing...")
            self.init_arch_baseline_solver()
            if self.arch_baseline_solver is None:
                print("[ERROR] arch_baseline_solver still None; cannot test satisfiability.")
                return False

        # Create a new solver in the same context as the baseline solver.
        cloned_solver = z3.Solver(ctx=self.arch_baseline_solver.ctx)

        # Explicitly re-add the full architecture constraints from arch_smt2_str.
        try:
            cloned_solver.append(*self.arch_baseline_solver.assertions())
        except Exception as e:
            print(f"[ERROR] Failed to parse arch_smt2_str in _test_chunk_satisfiability: {e}")
            return False

        # Add the patch declarations if available.
        patch_decl_text = "\n".join(sorted(self.patch_declarations))
        if patch_decl_text.strip():
            try:
                parsed_decls = z3.parse_smt2_string(
                    f"(set-logic QF_UF)\n{patch_decl_text}",
                    ctx=self.arch_baseline_solver.ctx
                )
                cloned_solver.add(parsed_decls)
            except Exception as e:
                print(f"[WARNING] Failed to add patch declarations: {e}")

        # Build the script for the given constraints.
        script = "\n".join([
            "(set-logic QF_UF)",
            patch_decl_text,
            *constraints
        ])
        try:
            parsed = z3.parse_smt2_string(script, ctx=self.arch_baseline_solver.ctx)
        except Exception as e:
            print(f"[ERROR] Failed to parse chunk constraints: {e}")
            return False

        cloned_solver.add(parsed)

        return (cloned_solver.check() == z3.sat)

    def generate_repaired_configs(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        constraints = []

        print("\n[INFO] Generating repaired configuration files...\n")

        # Parse architecture constraints if we haven't already
        if not self.parsed_arch_constraints:
            print("[ERROR] Parsed arch constraints are empty! Check parsing step.")

        # Get approximate constraints from existing config
        approx_constraints_raw = Klocalizer.get_config_file_constraints(self.existing_config_path)
        print(f"[DEBUG] Loaded {len(approx_constraints_raw)} approximate constraints from config")

        # Convert approximate constraints to the architecture context
        approx_constraints = []
        for constraint in approx_constraints_raw:
            try:
                # Convert the constraint to the architecture context
                if constraint.ctx == self.arch_ctx:
                    approx_constraints.append(constraint)
                else:
                    # Parse the constraint in the architecture context
                    constraint_str = constraint.sexpr()
                    # We need to wrap it in a proper SMT-LIB script
                    smt_script = "(set-logic QF_UF)\n"
                    # Extract variable names from the constraint
                    var_names = re.findall(r'CONFIG_[A-Za-z0-9_]+', constraint_str)
                    for var in var_names:
                        smt_script += f"(declare-const {var} Bool)\n"
                    smt_script += f"(assert {constraint_str})\n"

                    try:
                        parsed = z3.parse_smt2_string(smt_script, ctx=self.arch_ctx)
                        if parsed:
                            approx_constraints.append(parsed[0])
                    except Exception as e:
                        print(f"[WARNING] Failed to parse constraint in architecture context: {e}")
                        # Fallback: try to create a direct BoolVal
                        try:
                            if constraint.is_true():
                                approx_constraints.append(z3.BoolVal(True, self.arch_ctx))
                            elif constraint.is_false():
                                approx_constraints.append(z3.BoolVal(False, self.arch_ctx))
                            else:
                                # Skip this constraint as we can't convert it properly
                                print(f"[WARNING] Skipping complex constraint: {constraint_str}")
                        except:
                            print(f"[WARNING] Skipping constraint that couldn't be converted: {constraint_str}")
            except Exception as e:
                print(f"[WARNING] Error processing constraint: {e}")

        print(f"[DEBUG] Converted {len(approx_constraints)} approximate constraints to architecture context")

        groups = self.merged_groups if hasattr(self, "merged_groups") and self.merged_groups else {1: self.patch_constraints}

        for group_id, constraints in sorted(groups.items()):
            print(f"\n[INFO] Processing constraint group {group_id}...")

            try:
                # Extract all CONFIG variables from constraints
                config_vars = set()
                for constraint in constraints:
                    matches = re.findall(r'CONFIG_[A-Za-z0-9_]+', constraint)
                    config_vars.update(matches)

                # Build SMT script with declarations for all variables
                smt_script = "(set-logic QF_UF)\n"
                for var in config_vars:
                    smt_script += f"(declare-const {var} Bool)\n"

                # We'll parse the constraints in separate groups to maintain order

                arch_constraints = list(self.arch_baseline_solver.assertions())

                # Parse the patch constraints
                patch_smt = "(set-logic QF_UF)\n"
                for var in config_vars:
                    patch_smt += f"(declare-const {var} Bool)\n"
                patch_smt += "\n" + "\n".join(constraints)

                try:
                    parsed_patch_constraints = z3.parse_smt2_string(patch_smt, ctx=self.arch_ctx)
                    parsed_patch_constraints = [c for c in parsed_patch_constraints if isinstance(c, z3.BoolRef)]
                except Exception as e:
                    print(f"[ERROR] Failed to parse patch constraints: {str(e)}")
                    with open("failed_patch.smt2", "w") as f:
                        f.write(patch_smt)
                    raise

                # Write the combined SMT script to file for debugging
                smt_script_full = "(set-logic QF_UF)\n"

                # Combine constraints in the correct order
                full_constraints = arch_constraints + parsed_patch_constraints

                # First write the architecture constraints
                for c in arch_constraints:
                    smt_script_full += "(assert " + c.sexpr() + ")\n"

                # Finally write the patch constraints
                for c in parsed_patch_constraints:
                    smt_script_full += "(assert " + c.sexpr() + ")\n"

                # Check basic satisfiability without approximate constraints
                basic_solver = z3.Solver(ctx=self.arch_ctx)
                basic_solver.add(full_constraints)
                basic_sat = basic_solver.check() == z3.sat

                if basic_sat:
                    print(f"[INFO] Constraint group {group_id} is satisfiable without approximate constraints")

                    # Get a model directly from the basic solver
                    basic_model = basic_solver.model()
                    print(f"[DEBUG] Basic model declarations: {len(basic_model.decls())}")
                    print(f"[DEBUG] Basic model True assignments: {sum(1 for d in basic_model.decls() if basic_model[d] == True)}")

                else:
                    print(f"[WARNING] Constraint group {group_id} is UNSATISFIABLE without approximate constraints")

                    # Get the unsat core
                    unsat_core = basic_solver.unsat_core()
                    print(f"[DEBUG] Basic unsat core size: {len(unsat_core)}")
                    if unsat_core:
                        print(f"[DEBUG] First few core constraints: {[str(c) for c in list(unsat_core)[:5]]}")

                # Create model sampler with the full constraints
                model_sampler = Klocalizer.Z3ModelSampler(
                    full_constraints,
                    approximate_constraints=approx_constraints,
                    random_seed=None,
                    logger=None
                )

                is_sat, result = model_sampler.sample_model_with_ctx(self.arch_ctx)

                if is_sat:
                    model = result
                    print(f"[DEBUG] Model declarations: {len(model.decls())}")
                    print(f"[DEBUG] True assignments: {sum(1 for d in model.decls() if model[d])}")
                    print(f"[DEBUG] CONFIG_ assignments: {sum(1 for d in model.decls() if str(d).startswith('CONFIG_'))}")

                    config_count = sum(1 for d in model.decls() if str(d).startswith('CONFIG_'))
                    module_count = 0  # This might need to be calculated differently based on the original logic
                    print(f"[DEBUG] Total CONFIG_ declarations found in model: {config_count}")
                    print(f"[DEBUG] Total potential module options: {module_count}")

                    config_text = Klocalizer.get_config_from_model(
                        model=model,
                        arch=self.arch,
                        set_tristate_m=False,
                        allow_non_visibles=False,
                        approximate_config=self.existing_config_path  # File path as before.
                    )

                    config_lines = config_text.splitlines()
                    y_count = sum(1 for l in config_lines if '=y' in l)
                    m_count = sum(1 for l in config_lines if '=m' in l)
                    print(f"[DEBUG] Config breakdown:")
                    print(f"  y (built-in): {y_count}")
                    print(f"  m (module): {m_count}")
                    print(f"[DEBUG] Total lines to write: {len(config_lines)}")
                    config_options = [l for l in config_lines if l.startswith('CONFIG_')]
                    print(f"[DEBUG] Total CONFIG options to write: {len(config_options)}")

                    config_filename = os.path.join(output_dir, f"repaired_config_{group_id}.config")
                    with open(config_filename, "w") as f:
                        f.write(config_text)
                    print(f"[SUCCESS] Generated repaired config: {config_filename}")
                else:
                    print(f"[WARNING] Constraint group {group_id} is UNSAT")
            except Exception as e:
                print(f"[ERROR] Failed processing group {group_id}: {str(e)}")
                continue

        return constraints

def process_complete_smt_script(
        full_script,            # chunk-level smt text (with declare-const, etc.)
        chunk_idx,
        start_index,
        local_constraints,      # the slice of constraints for this chunk
        all_constraints,
        arch_smt2_str,
        shared_data,
        patch_declarations,
        arch_name
):
    chunk_id = chunk_idx + 1
    print(f"\n=== Processing chunk {chunk_id} ===")

    valid_indexes = []
    temp_unsat = []  # store constraint strings that cause unsat

    tracked_constraints = {}  # maps label -> constraint string (for unsat core debugging)
    added_constraints = set() # set of labels already added

    try:
        # --- 1) parse arch constraints once into a baseline solver ---
        ctx = z3.Context()
        base_solver = z3.Solver(ctx=ctx)

        if arch_smt2_str and arch_smt2_str.strip():
            try:
                arch_exprs = z3.parse_smt2_string(arch_smt2_str, ctx=ctx)
                base_solver.add(arch_exprs)
                print(f"Debug: Parsed {len(arch_exprs)} arch constraints into baseline solver.")
            except Exception as e:
                print(f"Error parsing arch_smt2_str in worker: {e}")
                return [], [], chunk_id
        else:
            print("Debug: arch_smt2_str is empty, no arch constraints added.")

        # --- 1.1) Add extra arch-specific assertions in the worker ---
        # TODO: This is a temporary fix for x86_64
        if arch_name == "x86_64":
            # Define the list of other architectures to assert as "not present".
            other_archs = [
                "ALPHA", "ARC", "ARM", "ARM64", "C6X", "CSKY", "H8300", "HEXAGON", "IA64",
                "LOONGARCH", "M68K", "MICROBLAZE", "MIPS", "NDS32", "NIOS2", "OPENRISC",
                "PARISC", "PPC", "PPC32", "PPC64", "RISCV", "S390", "SPARC", "SPARC32",
                "SPARC64", "SUPERH", "SUPERH32", "SUPERH64", "UML", "UNICORE32", "XTENSA"
            ]

            # Build extra arch declarations and assertions as a single SMT2 script.
            extra_arch_str = "(set-logic QF_UF)\n" + "\n".join([
                "(declare-const CONFIG_X86 Bool)",
                "(declare-const CONFIG_X86_64 Bool)",
                "(declare-const CONFIG_X86_32 Bool)",
                "(declare-const BITS_64 Bool)",
                "(declare-const BITS_32 Bool)",
                "(declare-const CONFIG_BROKEN Bool)",
                "(assert CONFIG_X86)",
                "(assert CONFIG_X86_64)",
                "(assert (not CONFIG_X86_32))",
                "(assert BITS_64)",
                "(assert (not BITS_32))"
            ])

            # Add negative assertions for other architectures.
            for arch in other_archs:
                extra_arch_str += f"\n(declare-const CONFIG_{arch} Bool)"
                extra_arch_str += f"\n(assert (not CONFIG_{arch}))"

            # Finally, assert that CONFIG_BROKEN is not set.
            extra_arch_str += "\n(assert (not CONFIG_BROKEN))"

            try:
                extra_exprs = z3.parse_smt2_string(extra_arch_str, ctx=ctx)
                base_solver.add(extra_exprs)
                print(f"Debug: Added {len(extra_exprs)} extra arch-specific assertions in worker.")
            except Exception as e:
                print(f"Error adding extra arch assertions in worker: {e}")

        # --- 2) parse patch declarations from full_script and add to baseline ---
        declarations = []
        print("Debug: Parsing patch declarations...")
        declarations = sorted(patch_declarations)
        if declarations:
            decl_script = "\n".join(["(set-logic QF_UF)"] + declarations)
            try:
                parsed_decls = z3.parse_smt2_string(decl_script, ctx=ctx)
                base_solver.add(parsed_decls)
                print(f"Debug: Parsed {len(parsed_decls)} declarations.")
            except Exception as e:
                print(f"Declaration context error: {str(e)[:200]}")
                return [], [], chunk_id
        else:
            print("Debug: No patch declarations found.")

        # --- 3) process patch constraints one-by-one ---
        print(f"Debug: Processing {len(local_constraints)} patch constraints...")
        for idx, constraint_str in enumerate(local_constraints):
            print(f"Processing constraint {idx}: {constraint_str.strip()}")
            current_index = start_index + idx

            # update shared counter
            if shared_data is not None:
                shared_data.counter += 1
                if (idx + 1) % 10 == 0:  # debug print every 10 constraints
                    print(f"Worker {os.getpid()} processed {shared_data.counter} constraints.")

            try:
                # create a new solver by cloning the baseline
                current_solver = z3.Solver(ctx=ctx)
                current_solver.append(*base_solver.assertions())
                added_constraints.clear()  # clear the set

                print(f"Valid indexes: {valid_indexes}")  # Are there any?
                print(f"Added constraints: {added_constraints}")  # What's already in here?

                # add all previously valid patch constraints
                for pv_idx in valid_indexes:
                    label = f"patch_{pv_idx}"
                    print(f"Checking label {label}")
                    if label not in added_constraints:
                        pv_constraint = local_constraints[pv_idx - start_index]
                        print(f"Adding previous constraint {pv_idx}: {pv_constraint}")
                        pv_script = "\n".join([
                            "(set-logic QF_UF)",
                            *declarations,
                            pv_constraint
                        ])
                        parsed_pv = z3.parse_smt2_string(pv_script, ctx=ctx)
                        if parsed_pv:
                            current_solver.assert_and_track(parsed_pv[-1], label)
                            print(f"Added constraint {label}")
                            tracked_constraints[label] = f"[PATCH idx={pv_idx}] {pv_constraint}"
                            added_constraints.add(label)

                # add the new constraint under test
                fail_label = f"patch_{current_index}"
                constraint_script = "\n".join([
                    "(set-logic QF_UF)",
                    *declarations,
                    constraint_str
                ])
                parsed = z3.parse_smt2_string(constraint_script, ctx=ctx)
                if parsed:
                    # Add the constraint directly
                    constraint_expr = parsed[-1]
                    current_solver.assert_and_track(constraint_expr, fail_label)
                    tracked_constraints[fail_label] = f"[PATCH idx={current_index}] {constraint_str}"
                    added_constraints.add(fail_label)
                else:
                    print(f"Empty parse result for constraint {idx}, skipping.")
                    continue

                # check sat
                assertions = current_solver.assertions()
                print(f"Total assertions: {len(assertions)}")
                # Print just the last few (non-arch) assertions
                last_n = 5  # or however many you want to see
                print("Last added assertions:")
                for assertion in assertions[-last_n:]:
                    print(f"  {assertion}")
                result = current_solver.check()
                if result == z3.sat:
                    valid_indexes.append(current_index)
                    print(f"Constraint {idx} is satisfiable")
                elif result == z3.unsat:
                    print(f"Constraint {idx} causes unsat:")
                    print(f"  - {constraint_str.strip()}")

                    unsat_solver = z3.Solver(ctx=ctx)
                    unsat_solver.set("unsat_core", True)
                    unsat_solver.append(*base_solver.assertions())

                    # add previously valid constraints with tracking
                    for pv_idx in valid_indexes:
                        label = f"patch_{pv_idx}"
                        if label not in added_constraints:
                            pv_constraint = local_constraints[pv_idx - start_index]
                            pv_script = "\n".join([
                                "(set-logic QF_UF)",
                                *declarations,
                                pv_constraint
                            ])
                            parsed_pv = z3.parse_smt2_string(pv_script, ctx=ctx)
                            if parsed_pv:
                                unsat_solver.assert_and_track(parsed_pv[-1], label)
                                tracked_constraints[label] = f"[PATCH idx={pv_idx}] {pv_constraint}"
                                added_constraints.add(label)

                    if fail_label not in added_constraints:
                        parsed_fail = z3.parse_smt2_string(constraint_script, ctx=ctx)
                        if parsed_fail:
                            unsat_solver.assert_and_track(parsed_fail[-1], fail_label)
                            tracked_constraints[fail_label] = f"[PATCH idx={current_index}] {constraint_str}"
                            added_constraints.add(fail_label)

                    if unsat_solver.check() == z3.unsat:
                        unsat_core = unsat_solver.unsat_core()
                        print(f"  - Raw unsat core: {unsat_core}")
                        for core_constraint in unsat_core:
                            c_label = str(core_constraint)
                            if c_label in tracked_constraints:
                                print(f"    - {tracked_constraints[c_label]}")
                            else:
                                print(f"    - Warning: {c_label} not found in tracked_constraints")

                    print(f"Adding to temp_unsat: {constraint_str.strip()}")
                    temp_unsat.append(constraint_str)
                else:
                    print(f"Constraint {idx} returned UNKNOWN: {constraint_str.strip()}")

            except Exception as e:
                print(f"Constraint parse error ({idx}): {str(e)[:200]}")
                temp_unsat.append(constraint_str)
                continue

        print(f"Processed {len(local_constraints)} constraints in chunk {chunk_id}.")
        return valid_indexes, temp_unsat, chunk_id

    except Exception as e:
        print(f"Critical failure: {str(e)[:200]}")
        return [], [], chunk_id


def main():

    linux_ksrc = "/home/alexei/LinuxKernels/krepair_alg/initial_testing/linux_other300commitset_copy"
    existing_config_file = f"{linux_ksrc}/.config"
    unbootable_options_file = "/home/alexei/LinuxKernels/krepair_alg/linux_set50copy/unbootable_options.txt"
    output_dir = f"{linux_ksrc}/repaired_configs"

    krepair = krepairDivQ(linux_ksrc, existing_config_path=existing_config_file)

    # Get arch constraints
    krepair.get_arch_constraints("x86_64")
    krepair.get_complex_arch_constraints("x86_64", f"{linux_ksrc}/arch_constraints_x86_64.txt")

    start_time = time.time()

    # Read kextract output
    with open(f"{linux_ksrc}/x86_64_formulas.pkl/kextract", "r") as f:
        content = f.read()

    krepair.process_kconfig_dependencies(content)
    krepair.parse_patch_configs_file(f"{linux_ksrc}/patch_constraints.txt")
    krepair.remove_unbootable_options(unbootable_options_file)

    krepair.check_constraints_until_unsat_parallel()

    # Generate repaired config files
    krepair.generate_repaired_configs(output_dir)

    elapsed_time = time.time() - start_time
    print(f"Execution completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()