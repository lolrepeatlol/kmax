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
from collections import defaultdict, Counter
from concurrent.futures import as_completed, ProcessPoolExecutor

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("krepairDC")

DEVNULL = subprocess.DEVNULL

def get_arch_formulas_dir(formulas: str, arch: str) -> str:
    """Helper to construct the formulas directory path for a given architecture."""
    assert arch is not None, "Arch name cannot be None"
    return os.path.join(formulas, f"{arch}_formulas.pkl")

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

    return declarations, assertions

class krepairDC:
    DECL_PATTERN = re.compile(r"(CONFIG_[A-Z0-9_]+)")  # compile once for efficiency

    def __init__(self, linux_ksrc: str, existing_config_path: str):
        self.linux_ksrc = linux_ksrc  # Path to the Linux kernel source directory
        self.arch_smt2_str = ""
        self.arch_baseline_solver = None  # Solver for architecture-specific constraints

        self.dependency_graph = nx.DiGraph()
        self.unbootable_options = set() # about to be axed

        self.patch_constraints = []  # List of patch constraints (SMT2 strings)
        self.patch_declarations = set()  # Set of SMT2 declarations for configuration symbols
        self.unit_constraints = defaultdict(list)  # Maps unit names to lists of patch constraints

        self.merged_groups = {}  # Merged groups of constraints
        self.existing_config_path = existing_config_path  # Path to the existing .config file
        self.existing_config_constraints = []  # Stores constraints from .config
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

    def get_complex_arch_constraints(self, arch_name: str = "x86_64"):
        """
        Build the SMT‑LIB text for all arch constraints and store it in
        self.arch_smt2_str, then initialize self.arch_baseline_solver.
        """

        try:
            # 1) Ensure kclause is present / generate if missing
            self.arch = Arch(
                arch_name,
                linux_ksrc=self.linux_ksrc,
                arch_dir=get_arch_formulas_dir(self.linux_ksrc, arch_name),
                is_kclause_composite=True,
                kextract_version="next-20210426",
                loggerLevel=logging.INFO
            )
            kclause_path = os.path.join(self.linux_ksrc,
                                        f"{arch_name}_formulas.pkl", "kclause")
            if not os.path.exists(kclause_path) or os.path.getsize(kclause_path) == 0:
                print("Kclause file missing or empty, generating…")
                self.arch.generate_kclause()
            with open(kclause_path, "rb") as f:
                raw_constraints = pickle.load(f)

            # 2) Batch‑process into declarations/assertions
            items = list(raw_constraints.items())
            cpu_count = mp.cpu_count()
            chunk_size = max(1, len(items) // (cpu_count * 2))
            batches = [items[i:i + chunk_size]
                       for i in range(0, len(items), chunk_size)]
            with mp.Pool(processes=cpu_count) as pool:
                batch_results = pool.starmap(
                    partial(process_constraint_batch),
                    zip(batches, range(0, len(items), chunk_size))
                )

            all_decls, all_asserts = set(), []
            for decls, asserts in batch_results:
                all_decls.update(decls)
                all_asserts.extend(asserts)

            # 3) Assemble SMT‑LIB text
            smt = ["(set-logic QF_UF)"]
            smt.extend(sorted(all_decls))
            smt.extend(all_asserts)
            self.arch_smt2_str = "\n".join(smt)

            # 4) Initialize the solver once
            self.init_arch_baseline_solver()

        except Exception as e:
            print(f"Error in get_complex_arch_constraints: {e}")
            self.arch_smt2_str = ""

    def init_arch_baseline_solver(self):
        try:
            # Build one dedicated context/solver for arch constraints
            self.arch_ctx = z3.Context()
            self.arch_baseline_solver = z3.Solver(ctx=self.arch_ctx)

            if not (self.arch_smt2_str and self.arch_smt2_str.strip()):
                print("[debug] arch_smt2_str is empty; arch baseline solver not initialized")
                return

            # 1) whatever constraints you already had, parsed into self.arch_ctx
            arch_exprs = list(
                z3.parse_smt2_string(self.arch_smt2_str, ctx=self.arch_ctx)
            )

            # 2) pull the per‑arch extras (they are in the *default* ctx)
            extra_exprs = [
                e.translate(self.arch_ctx)                # copy into the right ctx
                for e in self.arch.get_arch_specific_constraints()
            ]
            arch_exprs.extend(extra_exprs)

            # 3) add everything
            self.arch_baseline_solver.add(arch_exprs)
            print(f"[debug] arch baseline solver initialized with {len(arch_exprs)} constraints")

            # 4) add Not(CONFIG_BROKEN) constraint
            CONFIG_BROKEN = z3.Bool("CONFIG_BROKEN", ctx=self.arch_ctx)
            self.arch_baseline_solver.add(z3.Not(CONFIG_BROKEN))

        except Exception as e:
            print(f"[error] initializing arch baseline solver: {e}")
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

        seen = set()
        self.patch_constraints = []
        self.unit_constraints = defaultdict(list)
        self.unit_configs = defaultdict(set)

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
            if assertion_str not in seen:
                seen.add(assertion_str)
                self.patch_constraints.append(assertion_str)
                self.unit_constraints[current_unit].append(assertion_str)
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

        # Create a separate unit for constraints containing "not"
        not_constraints = []
        new_unit_constraints = defaultdict(list)

        for unit, constraints in self.unit_constraints.items():
            for constraint in constraints:
                if "not" in constraint:
                    not_constraints.append(constraint)  # Store separately
                else:
                    new_unit_constraints[unit].append(constraint)  # Keep in original unit

        # Assign the modified unit constraints back
        self.unit_constraints = new_unit_constraints

        # Add the new unit for "not" constraints if any exist
        if not_constraints:
            self.unit_constraints["not_constraints"] = not_constraints

        # Also reorder self.patch_constraints
        not_patch_constraints = [c for c in self.patch_constraints if "not" in c]
        self.patch_constraints = [c for c in self.patch_constraints if "not" not in c] + not_patch_constraints

        # Debugging Output (Optional)
        print("\n[Debug] Reordered Unit Constraints:")
        for unit, constraints in self.unit_constraints.items():
            print(f"Unit: {unit}")
            for c in constraints:
                print(f"  {c}")

        print("\n[Debug] Reordered Patch Constraints:")
        for c in self.patch_constraints:
            print(c)

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

    def check_constraints_until_unsat_parallel(self, num_processes=24):
        """
        Main function that performs parallel SMT-based satisfiability checks on
        self.patch_constraints, grouping by compilation unit and merging results.
        """

        def get_sorted_units():
            """
            Gather and sort compilation units by their number of constraints.

            :returns: List of (unit_name, constraints_list) sorted descending by list length.
            """
            units = list(self.unit_constraints.items())
            units.sort(key=lambda x: len(x[1]), reverse=True)
            return units

        def gather_unique_constraints(units):
            """
            Build a deduplicated list of all constraints in unit order.
            """
            unique = []
            seen = set()
            for unit_name, constraints in units:
                for c in constraints:
                    if c not in seen:
                        unique.append(c)
                        seen.add(c)
            return unique

        def determine_num_chunks(total_unique, num_threads):
            """
            Decide how many chunks to split into based on total constraints.
            """
            if total_unique < 50:
                return 3
            elif total_unique < 1000:
                return min(12, num_threads)
            else:
                return min(num_threads, max(8, total_unique // 150))

        def distribute_constraints(unique_constraints, num_chunks):
            """
            Evenly split the unique constraints list into num_chunks parts.
            """
            total = len(unique_constraints)
            base, extra = divmod(total, num_chunks)
            chunks, indexes_by_chunk = [], []
            idx = 0
            for i in range(num_chunks):
                size = base + 1 if i < extra else base
                chunk = unique_constraints[idx:idx+size]
                chunks.append(chunk)
                indexes_by_chunk.append(list(range(idx, idx+size)))
                idx += size

            print(f"Final chunk sizes: {[len(chunk) for chunk in chunks]}")
            for i, idxs in enumerate(indexes_by_chunk):
                print(f"Chunk {i} global indexes: {idxs}")

            return chunks, indexes_by_chunk

        def detect_duplicates(chunks):
            """
            Sanity check: warn if any normalized constraint appears
            in more than one chunk.
            """
            def normalize(c): return " ".join(c.strip().split())
            flat = [normalize(c) for chunk in chunks for c in chunk]
            dups = [c for c, cnt in Counter(flat).items() if cnt > 1]
            if dups:
                print(f" WARNING: Detected {len(dups)} duplicate constraints!")
                for d in dups[:10]:
                    print(f"  - {d}")

        def build_smt_scripts(chunks):
            """
            Assemble the full SMT-LIB script for each chunk.
            """
            header = "\n".join(sorted(self.patch_declarations))
            template = f"""(set-logic QF_UF)
    {header}
    %CONSTRAINTS%
    (check-sat)
    """
            return [
                template.replace("%CONSTRAINTS%", "\n".join(chunk))
                for chunk in chunks
            ]

        def execute_parallel(scripts, chunks, indexes_by_chunk):
            """
            Run each SMT script in parallel, tracking progress and collecting results.
            """
            manager = mp.Manager()
            shared = manager.Namespace()
            shared.counter = 0

            total_constraints = sum(len(chunk) for chunk in chunks)
            pbar = tqdm(total=total_constraints, desc="Processing constraints")
            filtered = [[] for _ in scripts]
            results = [None] * len(scripts)
            all_temp_unsat = {}

            with ProcessPoolExecutor(max_workers=num_processes) as ex:
                futures = {
                    ex.submit(
                        process_complete_smt_script,
                        scripts[i],
                        i,
                        indexes_by_chunk[i],
                        chunks[i],
                        self.patch_constraints,
                        self.arch_smt2_str,
                        shared,
                        self.patch_declarations,
                        self.arch.name
                    ): i for i in range(len(scripts))
                }

                while any(not f.done() for f in futures):
                    pbar.n = shared.counter
                    pbar.refresh()
                    time.sleep(0.5)
                pbar.n = shared.counter
                pbar.refresh()
                pbar.close()

                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        valid_idx, temp_unsat_list, chunk_id, local_final = fut.result()
                        print(f"DEBUG: Result from chunk {i} - Valid global indexes: {valid_idx}")
                        results[i] = valid_idx
                        filtered[i] = local_final
                        if temp_unsat_list:
                            print(f"DEBUG: Temp UNSAT from chunk {i}: {temp_unsat_list}")
                            all_temp_unsat[chunk_id] = temp_unsat_list
                    except Exception as e:
                        print(f"Error in chunk {i}: {e}")

            return filtered, results, all_temp_unsat

        def collect_valid_indexes(results_by_index):
            """
            Flatten and log all valid global indexes from worker results.
            """
            all_valid = []
            for i, res in enumerate(results_by_index):
                print(f"DEBUG: Chunk {i+1}/{len(results_by_index)} valid global indexes: {res}")
                if res:
                    all_valid.extend(res)
            print(f"DEBUG: Total valid global indexes: {len(all_valid)}")
            return all_valid

        def print_filtered_chunks(filtered_chunks):
            """
            Print each worker’s post-filtered constraint list.
            """
            print("\n=== Filtered Constraints from Each Worker (No TempUnsat) ===")
            for i, chunk in enumerate(filtered_chunks):
                print(f"Filtered Chunk {i}: {len(chunk)} constraints")
                for c in chunk:
                    print(f"  - {c.strip()}")

        def assemble_final_constraints(valid_by_chunk):
            """
            Concatenate all remaining constraints in chunk order.
            """
            final = []
            for cid in sorted(valid_by_chunk):
                final.extend(valid_by_chunk[cid])
            return final

        def write_final_chunks(valid_by_chunk):
            """
            Log final groups and write each to a .smt2 file.
            """
            print("\n--- Final Grouped Constraints ---")
            for cid, cons in sorted(valid_by_chunk.items()):
                print(f"\nChunk {cid}: {len(cons)} constraints")
                for c in cons[:5]:
                    print(f"  - {c.strip()}")
                fname = f"final_chunk_{cid}.smt2"
                with open(fname, "w") as f:
                    f.write("\n".join(cons))

        def update_patch_constraints(all_valid_indexes):
            """
            Filter self.patch_constraints to keep only those at valid indexes.
            """
            max_valid_index = len(self.patch_constraints) - 1
            filtered = [i for i in all_valid_indexes if i <= max_valid_index]
            self.patch_constraints = [self.patch_constraints[i] for i in filtered]


        def finalize_results(valid_by_chunk, never_sat, all_temp_unsat):
            """
            Update self.patch_constraints with final valid constraints and print a summary.

            Prints:
              - Total valid constraints
              - Total temporarily unsatisfiable (temp_unsat) constraints
              - Total and list of permanently unsatisfiable (never_sat) constraints
            """
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


        # Sort and group constraints into chunks
        units = get_sorted_units()
        unique_constraints = gather_unique_constraints(units)
        num_chunks = determine_num_chunks(len(unique_constraints), num_processes)
        print(f"Distributing {len(unique_constraints)} unique constraints into {num_chunks} chunks.")
        chunks, global_indexes = distribute_constraints(unique_constraints, num_chunks)
        detect_duplicates(chunks)  # Sanity check: Duplicates
        scripts = build_smt_scripts(chunks)  # Build SMT scripts for each chunk
        filtered_chunks, results_by_index, all_temp_unsat = execute_parallel(scripts, chunks, global_indexes)  # Test constraints in groups

        # Retrieve and print all valid constraints from results
        all_valid = collect_valid_indexes(results_by_index)
        update_patch_constraints(all_valid)
        print(f"Successfully processed {len(self.patch_constraints)} constraints")
        print_filtered_chunks(filtered_chunks)

        # Merge chunks and process TempUnsat constraints
        valid_by_chunk = {i: fc for i, fc in enumerate(filtered_chunks) if fc}
        valid_by_chunk = self._attempt_merge_chunks(valid_by_chunk, first_phase_only=True)
        never_sat = self._process_temp_unsat(all_temp_unsat, valid_by_chunk)
        finalize_results(valid_by_chunk, never_sat, all_temp_unsat)

        # Final merge pass to ensure all constraints are grouped tightly
        valid_by_chunk = self._attempt_merge_chunks(valid_by_chunk)
        self.patch_constraints = assemble_final_constraints(valid_by_chunk)
        print(f"Final constraints count after merging: {len(self.patch_constraints)}")
        write_final_chunks(valid_by_chunk)

        # Store the final merged groups for future use
        self.merged_groups = valid_by_chunk
        return {
            "added_constraints": len(self.patch_constraints),
            "temp_unsat": all_temp_unsat,
            "never_sat": never_sat
        }

    def _process_temp_unsat(self, all_temp_unsat, valid_by_chunk):
        """
        Processing for temp_unsat constraints.
        For each group in valid_by_chunk we create one solver instance.
        For each candidate temp_unsat constraint we push a new context, add it (with its full
        declarations) to the group’s solver, and test for satisfiability.
        If it is sat, we pop and then permanently add it (updating the group's declared tokens).
        If unsat, we pop, increment a strike, and once the candidate has reached the maximum
        allowed attempts (3 if there are 3+ groups, or equal to the number of groups if fewer),
        it is marked as never_sat.
        """
        never_sat = set()

        # Combine all temp_unsat constraints into a unique list.
        combined_temp_unsat = set()
        for constraints in all_temp_unsat.values():
            combined_temp_unsat.update(constraints)
        combined_temp_unsat = list(combined_temp_unsat)

        # Initialize strike counts.
        strike_counts = {c: 0 for c in combined_temp_unsat}
        group_count = len(valid_by_chunk)
        max_attempts = 3 if group_count >= 3 else group_count

        # Helper: Given a candidate constraint and the set of tokens already declared for the group,
        # build an SMT2 snippet that declares the union of all tokens (group tokens and candidate tokens)
        # and then asserts the candidate constraint.
        # Returns the parsed expressions and the candidate's tokens (to update the group's declared set later).
        def parse_candidate_constraint(constraint, group_declared_tokens):
            candidate_tokens = set(re.findall(r"(CONFIG_[A-Z0-9_]+)", constraint))
            all_tokens = group_declared_tokens.union(candidate_tokens)
            decl_lines = "\n".join(f"(declare-const {t} Bool)" for t in all_tokens)
            full_script = f"(set-logic QF_UF)\n{decl_lines}\n{constraint}"
            exprs = z3.parse_smt2_string(full_script, ctx=self.arch_baseline_solver.ctx)
            return exprs, candidate_tokens

        # Create one solver per group and track declared tokens for each group.
        solvers = {}
        solver_declared_tokens = {}
        for group_id, constraints in valid_by_chunk.items():
            solver = z3.Solver(ctx=self.arch_baseline_solver.ctx)
            # Add baseline assertions.
            solver.append(*self.arch_baseline_solver.assertions())
            declared = set()
            for cons in constraints:
                # For each valid constraint, build a full SMT2 snippet that includes all needed declarations.
                exprs, tokens = parse_candidate_constraint(cons, declared)
                solver.add(*exprs)
                declared.update(tokens)
            solvers[group_id] = solver
            solver_declared_tokens[group_id] = declared

        # Process groups in order of increasing size.
        sorted_group_ids = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
        for group_id in sorted_group_ids:
            solver = solvers[group_id]
            print(f"Processing temp_unsat constraints for group {group_id} (current size: {len(valid_by_chunk[group_id])})")
            # Iterate over a copy of the remaining candidate constraints.
            for constraint in list(strike_counts.keys()):
                solver.push()
                # Build the SMT2 snippet for the candidate using the union of declared tokens and candidate tokens.
                exprs, candidate_tokens = parse_candidate_constraint(constraint, solver_declared_tokens[group_id])
                solver.add(*exprs)
                result = solver.check()
                if result == z3.sat:
                    solver.pop()  # Remove the temporary addition.
                    # Permanently add the candidate constraint.
                    exprs, candidate_tokens = parse_candidate_constraint(constraint, solver_declared_tokens[group_id])
                    solver.add(*exprs)
                    # Update the group's declared tokens.
                    solver_declared_tokens[group_id].update(candidate_tokens)
                    valid_by_chunk[group_id].append(constraint)
                    # Explicitly remove from tempunsat
                    combined_temp_unsat.remove(constraint)
                    print(f"  ✓ Added constraint to group {group_id}: {constraint.strip()}")
                    del strike_counts[constraint]
                else:
                    solver.pop()  # Revert the temporary addition.
                    strike_counts[constraint] += 1
                    print(f"  × Constraint failed for group {group_id}: {constraint.strip()} (strike {strike_counts[constraint]})")
                    if strike_counts[constraint] >= max_attempts:
                        never_sat.add(constraint)
                        print(f"  → Marked as never_sat: {constraint.strip()}")
                        del strike_counts[constraint]

            # If all temp_unsat constraints have been handled, exit early.
            if not strike_counts:
                break

        print(f"\nDEBUG: Final never_sat constraints count: {len(never_sat)}")
        return never_sat

    def _get_unsat_core(self, constraints):
        """
        Attempts to extract an unsatisfiable core from a list of SMT constraints.

        For each constraint, a fresh Boolean assumption (e.g. ``a0``, ``a1``, ...) is declared,
        and the constraint is rewritten as an implication: ``(=> ai <constraint>)``. A solver
        is created with unsat core tracking enabled, and the function checks satisfiability
        under these assumptions.

        If the constraints are unsatisfiable, the solver returns an unsat core containing
        some of the assumptions. This function then maps each assumption in the core
        back to the list of ``CONFIG_*`` options that appeared in the corresponding constraint.
        """
        # TODO: implement much simpler & more robust unsat core extraction

        def strip_outer_assert(constraint_text):
            """
            Removes an outer (assert …) wrapper if present.
            This simple function assumes a well-formed single assertion.
            """
            s = constraint_text.strip()
            if s.startswith("(assert"):
                s = s[len("(assert"):].strip()
                if s and s[-1] == ")":
                    s = s[:-1].strip()
            return s

        # 1. Extract all CONFIG_* symbols from the constraints.
        config_ids = set()
        config_pattern = r"(CONFIG_[A-Z0-9_]+)"
        for ct in constraints:
            config_ids.update(re.findall(config_pattern, ct))

        # 2. Gather user patch declarations (if available) and determine which CONFIG_* are already declared.
        declared_ids = set()
        patch_decl_text = ""
        if hasattr(self, 'patch_declarations') and self.patch_declarations:
            patch_decl_text = "\n".join(self.patch_declarations)
            declared_ids = set(re.findall(r"\(declare-const\s+([A-Z0-9_]+)\s+Bool\)", patch_decl_text))

        # 3. Auto-declare any missing CONFIG_* symbols.
        auto_decls = []
        for cfg in sorted(config_ids):
            if cfg not in declared_ids:
                auto_decls.append(f"(declare-const {cfg} Bool)")
        auto_decl_text = "\n".join(auto_decls)

        # 4. Build the SMT2 script.
        script_lines = []
        # Set logic and enable unsat core production.
        script_lines.append("(set-logic QF_UF)")
        script_lines.append("(set-option :produce-unsat-cores true)")
        if patch_decl_text:
            script_lines.append(patch_decl_text)
        if auto_decl_text:
            script_lines.append(auto_decl_text)

        # 5. For each constraint, declare a fresh Boolean assumption.
        num_constraints = len(constraints)
        for idx in range(num_constraints):
            script_lines.append(f"(declare-const a{idx} Bool)")

        # 6. Build a mapping from assumption name to config options in that constraint.
        constraint_config_map = {}
        for idx, ct in enumerate(constraints):
            inner = strip_outer_assert(ct)
            # Extract CONFIG_* options from the inner expression.
            config_options = re.findall(config_pattern, inner)
            constraint_config_map[f"a{idx}"] = config_options
            # Assert the implication using the assumption a{idx}.
            script_lines.append(f"(assert (=> a{idx} {inner}))")

        final_script = "\n".join(script_lines)
        # print("Final SMT2 script:\n", final_script)

        # 7. Create a fresh solver with unsat core tracking enabled.
        solver = z3.Solver(ctx=self.arch_baseline_solver.ctx)
        solver.set(unsat_core=True)
        try:
            solver.append(*self.arch_baseline_solver.assertions())
        except Exception as e:
            print(f"[ERROR] Failed to re-add baseline assertions: {e}")
            return None

        try:
            parsed = z3.parse_smt2_string(final_script, ctx=self.arch_baseline_solver.ctx)
            solver.add(parsed)
        except Exception as e:
            print(f"[ERROR] parse_smt2_string failed in get_unsat_core: {e}")
            return None

        # 8. Create a list of assumption literals using z3.Bool with the solver's context.
        assumption_literals = [z3.Bool(f"a{idx}", ctx=solver.ctx) for idx in range(num_constraints)]
        # 9. Check the solver with these assumptions (passed as positional arguments).
        res = solver.check(*assumption_literals)
        if res == z3.sat:
            print("[INFO] Unexpected: constraints are satisfiable when attempting to get unsat core.")
            return None
        elif res == z3.unknown:
            print("[WARNING] Solver returned unknown for unsat core check.")
            return None
        else:
            core = solver.unsat_core()
            # Build a mapping from unsat assumption names to config options.
            unsat_mapping = {}
            for item in core:
                item_name = str(item)
                unsat_mapping[item_name] = constraint_config_map.get(item_name, [])
            return unsat_mapping

    def _test_chunk_satisfiability(self, constraints):
        """
        Checks whether a given chunk of SMT constraints is satisfiable.

        This function creates a new Z3 solver using the same context as the
        architecture baseline solver. It re-adds the architecture-level constraints,
        patch declarations (if any), and then adds the given constraint chunk.
        It returns True if the resulting constraint set is satisfiable.
        """
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
            print(f"[ERROR] Failed to re-add baseline assertions in test_chunk_satisfiability: {e}")
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

    def _attempt_merge_chunks(self, valid_by_chunk, first_phase_only=False):
        """
        Attempts to merge satisfiable chunks of patch constraints.

        The merge process can run in one or two phases:

        1. Round-Robin Merge (Phase 1)
           Adjacent chunks (1 and 2, 3 and 4, etc.) are repeatedly merged until no more
           neighbouring pairs are satisfiable. This phase is executed only when
           ``first_phase_only`` is ``True``.

        2. Merge-As-Much-As-Possible (Phase 2)
           Remaining chunks are considered in size order and greedily merged whenever
           the combined constraints remain satisfiable. The loop stops when an
           iteration produces no successful merges.
           If first_phase_only is True, Phase 2 is limited to a single
           iteration (to allow later handling of temporarily-unsat constraints).
        """
        # TODO: split function

        tried_chunks = defaultdict(set)

        # PHASE 1: Repeated Round-Robin Merge (only if first_phase_only is True)
        if first_phase_only:
            print("\n=== Phase 1: Round-Robin Merge ===")
            round_robin_changed = True
            while round_robin_changed:
                round_robin_changed = False
                chunk_ids = sorted(valid_by_chunk.keys())
                if len(chunk_ids) < 2:
                    break  # nothing to merge if only one chunk remains.
                # Process adjacent pairs: (chunk_ids[0], chunk_ids[1]), (chunk_ids[2], chunk_ids[3]), etc.
                for idx in range(0, len(chunk_ids) - 1, 2):
                    c1 = chunk_ids[idx]
                    c2 = chunk_ids[idx + 1]
                    print(f"\nRound-robin pass: Trying to merge Chunk {c1} with Chunk {c2}...")
                    merged_constraints = valid_by_chunk[c1] + valid_by_chunk[c2]
                    if self._test_chunk_satisfiability(merged_constraints):
                        print(f"  ✓ Merge success: {c1} + {c2} → {len(merged_constraints)} constraints")
                        valid_by_chunk[c1] = merged_constraints
                        del valid_by_chunk[c2]
                        round_robin_changed = True
                    else:
                        unsat_core = self._get_unsat_core(merged_constraints)
                        print(f"  ✗ Merge failed: {c1} + {c2} → UNSAT; Unsat Core: {unsat_core}")
                if round_robin_changed:
                    print("At least one merge succeeded; re-running round-robin pass...")
            print("Round-robin merge phase completed.")
        else:
            print("Skipping Phase 1 (Round-Robin Merge) because first_phase_only is False.")

        # PHASE 2: Merge-As-Much-As-Possible (general merge phase)
        iteration = 0
        while True:
            chunk_ids = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
            if not chunk_ids:
                break

            iteration += 1
            if iteration > 100:
                print("Breaking merge loop after 100 iterations to prevent infinite loop.")
                break

            print(f"\n=== Iteration {iteration}: Merge As Much As Possible ===")
            print(f"Remaining chunks: {chunk_ids}")

            iteration_merged = False
            used_this_pass = set()

            for i, c1 in enumerate(chunk_ids):
                if c1 in used_this_pass:
                    continue
                for j in range(i + 1, len(chunk_ids)):
                    c2 = chunk_ids[j]
                    if c2 in used_this_pass or c2 in tried_chunks[c1]:
                        continue
                    print(f"\nPass: Trying to merge Chunk {c1} ({len(valid_by_chunk[c1])} constraints) "
                          f"with Chunk {c2} ({len(valid_by_chunk[c2])} constraints)...")
                    merged_constraints = valid_by_chunk[c1] + valid_by_chunk[c2]
                    if self._test_chunk_satisfiability(merged_constraints):
                        print(f"  ✓ Merge success: {c1} + {c2} → {len(merged_constraints)} constraints")
                        valid_by_chunk[c1] = merged_constraints
                        del valid_by_chunk[c2]
                        used_this_pass.add(c1)
                        used_this_pass.add(c2)
                        tried_chunks[c1].add(c2)
                        tried_chunks[c2].add(c1)
                        iteration_merged = True
                        break  # Exit inner loop to update ordering after a merge.
                    else:
                        unsat_core = self._get_unsat_core(merged_constraints)
                        print(f"  ✗ Merge failed: {c1} + {c2} → UNSAT; Unsat Core: {unsat_core}")
                        tried_chunks[c1].add(c2)
                        tried_chunks[c2].add(c1)
                if iteration_merged:
                    break

            # If first_phase_only is True, exit after the first successful iteration of Phase 2.
            if first_phase_only and iteration_merged:
                print("First iteration of merge-as-much-as-possible completed, stopping early for temp_unsat processing.")
                break

            if not iteration_merged:
                print("No merges succeeded in this iteration. Stopping merge-as-much-as-possible phase.")
                break
            else:
                print("Merges happened; starting a new iteration of general merging...")

        print("\nFinal Merging Completed.")
        final_chunks = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
        print(f"Remaining Chunks: {final_chunks}")
        return valid_by_chunk

    def generate_repaired_configs(self, output_dir: str):
        """
        Generates repaired Linux kernel configuration files for each constraint group
        and saves them to the specified output directory.

        For each constraint group, this function builds a combined SMT formula using:
        - architecture-specific constraints,
        - patch-specific constraints, and
        - approximate constraints derived from an existing ``.config`` file.

        It checks satisfiability of the combined formula using Z3. If satisfiable, a
        kernel configuration is generated from the model and saved to the output
        directory. Debug files, including the full SMT formula and model statistics,
        are also written for each group.
        """
        # TODO: break function into smaller parts & simplify

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        constraints = []

        print("\n[INFO] Generating repaired configuration files...\n")

        # Parse architecture constraints if we haven't already
        if not self.arch_baseline_solver.assertions():
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

                # Write full constraints to a file for debugging
                with open(f"full_constraints_group_{group_id}.smt2", "w") as f:
                    f.write(smt_script_full)

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
        full_script,
        chunk_idx,
        global_indexes,
        local_constraints,
        all_constraints,
        arch_smt2_str,
        shared_data,
        patch_declarations,
        arch_name
):
    """
    Filters a single chunk of patch constraints, keeping only those that remain
    satisfiable when added to the architecture baseline.

    For the given ``chunk_idx`` the function builds a fresh Z3 solver, loads
    architecture-wide constraints and patch-level declarations, then pushes each
    constraint one-by-one.  If adding a constraint preserves satisfiability it is
    kept; otherwise it is popped and recorded in temp_unsat.
    """
    chunk_id = chunk_idx + 1
    print(f"\n=== Processing chunk {chunk_id} ===")

    valid_indexes = []
    temp_unsat = []
    global_to_local = {g: i for i, g in enumerate(global_indexes)}

    try:
        # 1) Build a fresh context and solver
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)

        # 1.1) parse & add the baked‑in arch_smt2_str
        if arch_smt2_str and arch_smt2_str.strip():
            arch_exprs = z3.parse_smt2_string(arch_smt2_str, ctx=ctx)
            solver.add(arch_exprs)
            print(f"Debug: Parsed {len(arch_exprs)} arch constraints into baseline solver.")
        else:
            print("Debug: arch_smt2_str is empty, no arch constraints added.")

        # 1.2) pull in *all* per‑arch constraints via the helper
        # TODO: look into using existing baseline solver instead of pulling arch constraints again
        arch = Arch(arch_name)
        extra_arch_exprs = [
            expr.translate(ctx) for expr in arch.get_arch_specific_constraints()
        ]
        solver.add(extra_arch_exprs)
        solver.add(z3.Not(z3.Bool("CONFIG_BROKEN", ctx=ctx)))
        print(f"Debug: Added {len(extra_arch_exprs)} arch‑specific assertions from helper.")

        # 1.3) parse & add patch‑level declarations
        if patch_declarations:
            decl_script = "\n".join(["(set-logic QF_UF)"] + sorted(patch_declarations))
            parsed_decls = z3.parse_smt2_string(decl_script, ctx=ctx)
            solver.add(parsed_decls)
            print(f"Debug: Parsed {len(parsed_decls)} declarations.")
        else:
            print("Debug: No patch declarations found.")

        # 2) push/pop through each constraint
        print(f"Debug: Processing {len(local_constraints)} patch constraints...")
        for idx, constraint_str in enumerate(local_constraints):
            current_index = global_indexes[idx]
            if shared_data is not None:
                shared_data.counter += 1
                if (idx + 1) % 10 == 0:
                    print(f"Worker {os.getpid()} processed {shared_data.counter} constraints.")

            print(f"Processing constraint {idx}: {constraint_str.strip()}")
            solver.push()
            try:
                # include declarations so symbols are in scope
                script = "\n".join(["(set-logic QF_UF)"] +
                                   sorted(patch_declarations) +
                                   [constraint_str])
                parsed_c = z3.parse_smt2_string(script, ctx=ctx)
                if not parsed_c:
                    solver.pop()
                    continue

                solver.add(parsed_c)
                res = solver.check()
                if res == z3.sat:
                    valid_indexes.append(current_index)
                    print(f"Constraint {idx} is satisfiable, keeping it.")
                else:
                    solver.pop()
                    temp_unsat.append(constraint_str)
                    print(f"Constraint {idx} causes {res}, removing it.")
            except Exception as e:
                solver.pop()
                temp_unsat.append(constraint_str)
                print(f"Constraint error ({idx}): {e}")

        # 3) collect final lists
        final_local = []
        final_globals = []
        for gidx in valid_indexes:
            li = global_to_local[gidx]
            final_local.append(local_constraints[li])
            final_globals.append(gidx)

        print(f"Processed {len(local_constraints)} constraints in chunk {chunk_id}.")
        return final_globals, temp_unsat, chunk_id, final_local

    except Exception as e:
        print(f"Critical failure in chunk {chunk_id}: {e}")
        return [], [], chunk_id, []


def main():
    # Tester/prototyping function

    linux_ksrc = "/home/alexei/LinuxKernels/krepair_alg/pre-study-fixes/linux_copy"
    existing_config_file = f"{linux_ksrc}/.config"
    unbootable_options_file = "/home/alexei/LinuxKernels/krepair_alg/linux_set50copy/unbootable_options.txt"
    output_dir = f"{linux_ksrc}"

    krepair = krepairDC(linux_ksrc, existing_config_path=existing_config_file)

    # Get arch constraints
    krepair.get_complex_arch_constraints("x86_64")

    # Start recording amount of time for krepairDC mutex algorithm
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
    print(f"Algorithm 1 completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
