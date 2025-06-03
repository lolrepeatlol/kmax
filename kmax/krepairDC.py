import time
import z3
import re
from tqdm import tqdm
from typing import List, Dict, Tuple
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
    # CONFIG_FOO  |  BITS=32 / BITS=64  |  anything already pipe-quoted
    DECL_PATTERN = re.compile(r"(?:CONFIG_[A-Za-z0-9_]+|BITS=[0-9]+|\|[^|]+\|)")

    def __init__(self, linux_ksrc: str, existing_config_path: str):
        self.linux_ksrc = linux_ksrc  # Path to the Linux kernel source directory
        self.arch_smt2_str = ""
        self.arch_baseline_solver = None  # Solver for architecture-specific constraints

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
            print(f"[debug] Arch baseline solver initialized with {len(arch_exprs)} constraints")

            # 4) add Not(CONFIG_BROKEN) constraint
            CONFIG_BROKEN = z3.Bool("CONFIG_BROKEN", ctx=self.arch_ctx)
            self.arch_baseline_solver.add(z3.Not(CONFIG_BROKEN))

        except Exception as e:
            print(f"[error] Initializing arch baseline solver: {e}")
            self.arch_baseline_solver = None

    def is_arch_specific_unit(self, unit: str, target_arch: str, accept_x86: bool = True) -> bool:
        """
        Check if a unit is architecture-specific and doesn't match target architecture
        Returns True if unit should be skipped (not compatible with target_arch)
        """
        # TODO: update with get_archs_from_subdir
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
            found_tokens = self.DECL_PATTERN.findall(line)
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

        print("\n[Debug] Reordered Patch Constraints:")
        for c in self.patch_constraints:
            print(c)

    def get_patch_constraints(
            self,
            requirements: List[Tuple[str, List[int]]],
            kmax_constraints: Dict[str, List],
            line_constraints: Dict[str, Dict[str, Klocalizer.ConditionalBlock]],
            accumulated_constraints: List,
            arch
    ):
        """
        Collect the same constraints as original krepair, and
        store them in self.patch_constraints/self.unit_constraints.

        Constraints are deduplicated across units.

        requirements: list of (unit, [lines]) tuples
        kmax_constraints: mapping unit -> list of z3 constraint objects
        line_constraints: mapping srcfile -> { arch.name: ConditionalBlock }
        accumulated_constraints: list of z3 constraint objects applied everywhere
        arch: object with .name attribute (e.g. "x86_64")
        """

        seen = set()  # track every assertion we’ve already added
        self.patch_constraints = []
        self.unit_constraints = defaultdict(list)
        self.patch_declarations = set()  # <-- reset your declares here

        for unit, lines in requirements:
            # 1) skip directory‐only units
            if unit.endswith('/'):
                continue

            # 2) skip any unit that’s arch‐specific to another arch
            if self.is_arch_specific_unit(unit, arch.name, accept_x86=True):
                continue

            # 3) gather all z3 ExprRefs just like original
            srcfile = Klocalizer.unit2srcfile(unit)
            combined = list(kmax_constraints.get(unit, []))

            cb_map = line_constraints.get(srcfile, {})
            cb = cb_map.get(arch.name)
            if cb is not None:
                for line in lines:
                    if line == 0:
                        continue
                    deepest = cb.get_deepest_block(line)
                    if deepest:
                        combined.extend(deepest.pc.assertions())

            combined.extend(accumulated_constraints)

            # 4) for each constraint, serialize, declare all tokens, then dedup & store
            for c in combined:
                try:
                    sexpr = c.sexpr()
                except Exception as e:
                    logger.warning(f"Failed to serialize constraint for {unit}: {e}")
                    continue

                # collect tokens and add declare-const lines
                tokens = set(self.DECL_PATTERN.findall(sexpr))
                for t in sorted(tokens):
                    self.patch_declarations.add(f"(declare-const {t} Bool)")

                assertion = f"(assert {sexpr})"
                if assertion in seen:
                    continue  # skip duplicates across units
                seen.add(assertion)

                self.patch_constraints.append(assertion)
                self.unit_constraints[unit].append(assertion)

        total = sum(len(v) for v in self.unit_constraints.values())
        logger.info(f"Collected {total} unique patch constraints across {len(self.unit_constraints)} units")

    def check_constraints_until_unsat_parallel(self, num_processes=24):
        """
        Main function that performs parallel SMT-based satisfiability checks on
        self.patch_constraints, grouping by compilation unit and merging results.
        """

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
                return min(num_threads, max(8, total_unique // 100))

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
                        iteratively_test_constraints,
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


        units = list(self.unit_constraints.items())
        unique_constraints = gather_unique_constraints(units)
        num_chunks = determine_num_chunks(len(unique_constraints), num_processes)
        print(f"Distributing {len(unique_constraints)} unique constraints into {num_chunks} chunks.")
        chunks, global_indexes = distribute_constraints(unique_constraints, num_chunks)
        detect_duplicates(chunks)  # Sanity check: Duplicates
        scripts = build_smt_scripts(chunks)  # Build SMT scripts for each chunk
        filtered_chunks, results_by_index, all_temp_unsat = execute_parallel(scripts, chunks, global_indexes)  # Test constraints in groups

        # Collect and update the always‑sat constraints
        all_valid = collect_valid_indexes(results_by_index)
        update_patch_constraints(all_valid)
        print(f"Successfully processed {len(self.patch_constraints)} constraints")
        print_filtered_chunks(filtered_chunks)

        # Prepare the valid_by_chunk mapping
        valid_by_chunk = {i: fc for i, fc in enumerate(filtered_chunks) if fc}

        # Process ALL temp_unsat before merging ever happens
        never_sat = self._process_temp_unsat(all_temp_unsat, valid_by_chunk)
        finalize_results(valid_by_chunk, never_sat, all_temp_unsat)

        # Single‐shot merge: run both Phase 1 & Phase 2
        valid_by_chunk = self._attempt_merge_chunks(valid_by_chunk)

        # Assemble and write out final constraints
        self.patch_constraints = assemble_final_constraints(valid_by_chunk)
        print(f"Final constraints count after merging: {len(self.patch_constraints)}")
        write_final_chunks(valid_by_chunk)

        # Save & return
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
        If it is sat, we pop and then permanently add it (updating the group's declared tokens)
        and stop trying other groups.
        If it is unsat in every group, we then test it on its own (with only arch constraints).
        - If still unsat, mark as never_sat.
        - If sat, make a new group containing just that constraint.
        """
        # TODO: break into multiple functions
        never_sat = set()

        # Combine all temp_unsat constraints into a unique list.
        combined_temp_unsat = set()
        for constraints in all_temp_unsat.values():
            combined_temp_unsat.update(constraints)
        combined_temp_unsat = list(combined_temp_unsat)

        # Helper: Given a candidate constraint and the set of tokens already declared for the group,
        # build an SMT2 snippet that declares the union of all tokens (group tokens and candidate tokens)
        # and then asserts the candidate constraint.
        def parse_candidate_constraint(constraint, group_declared_tokens):
            candidate_tokens = set(self.DECL_PATTERN.findall(constraint))
            all_tokens = group_declared_tokens.union(candidate_tokens)
            decl_lines = "\n".join(f"(declare-const {t} Bool)" for t in all_tokens)
            full_script = f"(set-logic QF_UF)\n{decl_lines}\n{constraint}"
            exprs = z3.parse_smt2_string(full_script, ctx=self.arch_baseline_solver.ctx)
            return exprs, candidate_tokens

        # Build one solver per existing group
        solvers = {}
        solver_declared_tokens = {}
        for group_id, constraints in valid_by_chunk.items():
            solver = z3.Solver(ctx=self.arch_baseline_solver.ctx)
            solver.append(*self.arch_baseline_solver.assertions())
            declared = set()
            for cons in constraints:
                exprs, tokens = parse_candidate_constraint(cons, declared)
                solver.add(*exprs)
                declared.update(tokens)
            solvers[group_id] = solver
            solver_declared_tokens[group_id] = declared

        # Try groups in increasing-size order
        sorted_group_ids = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))

        for constraint in list(combined_temp_unsat):
            print(f"\nTrying to place constraint: {constraint.strip()}")
            placed = False

            # 1) Try to fit into any existing group
            for group_id in sorted_group_ids:
                solver = solvers[group_id]
                print(f" Processing group {group_id} (size {len(valid_by_chunk[group_id])})")
                solver.push()
                exprs, candidate_tokens = parse_candidate_constraint(constraint, solver_declared_tokens[group_id])
                solver.add(*exprs)
                if solver.check() == z3.sat:
                    solver.pop()
                    # permanently add to this group
                    exprs, candidate_tokens = parse_candidate_constraint(constraint, solver_declared_tokens[group_id])
                    solver.add(*exprs)
                    solver_declared_tokens[group_id].update(candidate_tokens)
                    valid_by_chunk[group_id].append(constraint)
                    combined_temp_unsat.remove(constraint)
                    print(f"  ✓ Added to group {group_id}")
                    placed = True
                    break
                else:
                    solver.pop()
                    print(f"  × Rejected by group {group_id}")

            # 2) If it didn't fit anywhere, try it on its own
            if not placed:
                print(f" Trying to place on its own")
                # build a fresh solver with only arch constraints
                solo_solver = z3.Solver(ctx=self.arch_baseline_solver.ctx)
                solo_solver.append(*self.arch_baseline_solver.assertions())
                exprs, tokens = parse_candidate_constraint(constraint, set())
                solo_solver.add(*exprs)

                if solo_solver.check() == z3.sat:
                    # create a new group for this single constraint
                    new_group_id = max(valid_by_chunk.keys(), default=-1) + 1
                    valid_by_chunk[new_group_id] = [constraint]
                    solvers[new_group_id] = solo_solver
                    solver_declared_tokens[new_group_id] = tokens
                    combined_temp_unsat.remove(constraint)
                    # re-sort groups so future constraints see the new group
                    sorted_group_ids = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
                    print(f"  + Created new group {new_group_id} for constraint")
                else:
                    never_sat.add(constraint)
                    combined_temp_unsat.remove(constraint)
                    print(f"  → Marked as never_sat: {constraint.strip()}")

        print(f"\nDEBUG: Final never_sat constraints count: {len(never_sat)}")
        return never_sat

    def _get_unsat_core(self, constraints):
        """
        Return { 'a<i>': [CONFIG_…] } for the constraints that
        participate in the unsat core, or None if the whole set is sat/unknown.
        """

        # TODO: look into skipping re-declaration pass (could cause edge cases)
        # 1. Collect CONFIG_* symbols we might need to declare
        all_cfg_ids = {cfg for ct in constraints
                       for cfg in self.DECL_PATTERN.findall(ct)}

        declared_ids = set()
        if self.patch_declarations:
            # Pull every SYMBOL from our existing “(declare-const SYMBOL Bool)” lines:
            declared_ids = {
                    tok
                    for d in self.patch_declarations
                    for tok in self.DECL_PATTERN.findall(d)
                }

        auto_decl_text = "\n".join(f"(declare-const {cfg} Bool)"
                                   for cfg in sorted(all_cfg_ids - declared_ids))

        header = "(set-logic QF_UF)\n" \
                 + "\n".join(self.patch_declarations) + "\n" \
                 + auto_decl_text + "\n"

        # 2. Build a fresh solver, copy arch baseline
        ctx    = self.arch_baseline_solver.ctx
        solver = z3.Solver(ctx=ctx)
        solver.set(unsat_core=True)
        solver.append(*self.arch_baseline_solver.assertions())

        # 3. Feed every constraint with assert-and-track
        tag_to_cfgs = {}
        for idx, ct in enumerate(constraints):
            tag = z3.Bool(f"a{idx}", ctx=ctx)
            expr = z3.parse_smt2_string(header + ct, ctx=ctx)[0]   # parse returns a list
            solver.assert_and_track(expr, tag)
            tag_to_cfgs[tag.decl().name()] = self.DECL_PATTERN.findall(ct)

        # 4. Check & extract core
        res = solver.check()
        if res != z3.unsat:
            print(f"[INFO]  set was {res}; unsat core not available")
            return None

        core = solver.unsat_core()               # list[BoolRef] returned by solver
        return {str(t): tag_to_cfgs[str(t)] for t in core}

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

    def _attempt_merge_chunks(self, valid_by_chunk):
        """
        Attempts to merge satisfiable chunks of patch constraints
        by merging as much as possible (greedily).

        Chunks are considered in size order and greedily merged whenever
        the combined constraints remain satisfiable. The loop stops when an
        iteration produces no successful merges.
        """

        tried_chunks = defaultdict(set)

        # Merge‑As‑Much‑As‑Possible (iterative size‑sorted greedy)
        print("\n=== Merge As Much As Possible ===")
        iteration = 0
        while True:
            chunk_ids = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
            iteration += 1
            if iteration > 100:
                print("Breaking merge loop after 100 iterations to prevent infinite loop.")
                break

            print(f"\n--- Iteration {iteration} ---")
            print(f"Remaining chunks (by size): {chunk_ids}")

            iteration_merged = False

            # try every pair in size order, break on first success
            for i, c1 in enumerate(chunk_ids):
                for c2 in chunk_ids[i+1:]:
                    if c2 in tried_chunks[c1]:
                        continue

                    print(f"\nPass: Trying to merge Chunk {c1} ({len(valid_by_chunk[c1])} constraints) "
                          f"with Chunk {c2} ({len(valid_by_chunk[c2])} constraints)...")
                    merged_constraints = valid_by_chunk[c1] + valid_by_chunk[c2]

                    if self._test_chunk_satisfiability(merged_constraints):
                        print(f"  ✓ Merge success: {c1} + {c2} → {len(merged_constraints)} constraints")
                        valid_by_chunk[c1] = merged_constraints
                        del valid_by_chunk[c2]
                        tried_chunks[c1].add(c2)
                        tried_chunks[c2].add(c1)
                        iteration_merged = True
                        break
                    else:
                        unsat_core = self._get_unsat_core(merged_constraints)
                        print(f"  ✗ Merge failed: {c1} + {c2} → UNSAT; Unsat Core: {unsat_core}")
                        tried_chunks[c1].add(c2)
                        tried_chunks[c2].add(c1)
                if iteration_merged:
                    # restart the outer while to re-sort sizes
                    break

            if not iteration_merged:
                print("No merges succeeded in this iteration. Stopping merge‑as‑much‑as‑possible phase.")
                break
            else:
                print("Merges happened; restarting size‑sorted merge...")

        print("\nFinal Merging Completed.")
        final_chunks = sorted(valid_by_chunk.keys(), key=lambda cid: len(valid_by_chunk[cid]))
        print(f"Remaining Chunks: {final_chunks}")
        return valid_by_chunk

    def generate_repaired_configs(self, output_dir: str, arch_name: str, approx_constraints_raw=None):
        """
        Generates repaired Linux kernel configuration files for each constraint group
        and saves them to the specified output directory.

        For each constraint group, this function builds a combined SMT formula using:
        - architecture-specific constraints,
        - patch-specific constraints, and
        - approximate constraints derived from an existing ``.config`` file.

        It checks satisfiability of the combined formula using Z3. If satisfiable, a
        kernel configuration is generated from the model and saved to the output
        directory.
        """
        # TODO: look into using _test_chunk_satisfiability() here instead?
        # TODO: clean up

        def _to_arch_ctx(exprs):
            """
            Translate every BoolRef in *exprs* into self.arch_ctx.
            Skip expressions that cannot be translated.
            """
            out = []
            for e in exprs:
                try:
                    out.append(e if e.ctx == self.arch_ctx else e.translate(self.arch_ctx))
                except z3.Z3Exception as err:
                    print(f"[WARN] skipping constraint that can’t translate: {err}")
            return out

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        constraints = []

        print("\n[INFO] Generating repaired configuration files...\n")

        # Check if architecture constraints are present
        assert self.arch_baseline_solver.assertions()

        # Only get approximate constraints if not provided (TODO: remove this)
        if approx_constraints_raw is None:
            # Get approximate constraints from existing config
            approx_constraints_raw = Klocalizer.get_config_file_constraints(self.existing_config_path)
            print(f"[DEBUG] Loaded {len(approx_constraints_raw)} approximate constraints from config")
        else:
            print(f"[DEBUG] Using {len(approx_constraints_raw)} provided approximate constraints")

        # Translate constraints to the architecture context
        approx_constraints = _to_arch_ctx(approx_constraints_raw)
        print(f"[DEBUG] Converted {len(approx_constraints)} approximate constraints")

        groups = self.merged_groups if hasattr(self, "merged_groups") and self.merged_groups else {1: self.patch_constraints}

        for group_id, constraints in sorted(groups.items()):
            print(f"\n[INFO] Processing constraint group {group_id}...")

            try:
                # Extract all CONFIG variables from constraints
                config_vars = set()
                for constraint in constraints:
                    matches = self.DECL_PATTERN.findall(constraint)
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

                # Create model sampler with the full constraints
                model_sampler = Klocalizer.Z3ModelSampler(
                    full_constraints,
                    approximate_constraints=approx_constraints,
                    ctx=self.arch_ctx,
                    random_seed=None,
                    logger=None
                )

                is_sat, result = model_sampler.sample_model()

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

                    config_filename = os.path.join(output_dir, f"{group_id}-{arch_name}.config")
                    with open(config_filename, "w") as f:
                        f.write(config_text)
                    print(f"[SUCCESS] Generated repaired config: {config_filename}")
                else:
                    print(f"[WARNING] Constraint group {group_id} is UNSAT")
            except Exception as e:
                print(f"[ERROR] Failed processing group {group_id}: {str(e)}")
                continue

        return constraints

def iteratively_test_constraints(
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

    linux_ksrc = "/home/alexei/LinuxKernels/krepair_alg/integration_testing/linux_copy_original"
    existing_config_file = f"{linux_ksrc}/.config"
    output_dir = f"{linux_ksrc}"

    krepair = krepairDC(linux_ksrc, existing_config_path=existing_config_file)

    # Get arch constraints
    krepair.get_complex_arch_constraints("x86_64")

    # Start recording amount of time for krepairDC mutex algorithm
    start_time = time.time()

    # Read kextract output
    with open(f"{linux_ksrc}/x86_64_formulas.pkl/kextract", "r") as f:
        content = f.read()

    krepair.parse_patch_configs_file(f"{linux_ksrc}/patch_constraints.txt")

    krepair.check_constraints_until_unsat_parallel()

    # Generate repaired config files
    krepair.generate_repaired_configs(output_dir, "x86_64")

    elapsed_time = time.time() - start_time
    print(f"Algorithm 1 completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()