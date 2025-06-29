import random
import re
import os
import csv
import math
import json
import shutil
import subprocess
import argparse
import statistics
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
sys.stdout.reconfigure(line_buffering=True)

# Constants
VENV_ACTIVATE = os.path.expanduser('/home/alexei/Miscellaneous/kmax/tester_venv/bin/activate')

def load_commits(path: str) -> List[str]:
    """
    Return the commit list exactly line-for-line.
    Blank / whitespace-only lines become an empty string ''.
    A warning is printed if any blanks are seen.
    """
    commits: List[str] = []
    blanks = 0

    with open(path, "r") as f:
        for line in f:
            sha = line.strip()
            if not sha:
                blanks += 1
            commits.append(sha)          # keep even the blank

    if len(commits) == 0 or len(commits) == blanks:
        raise RuntimeError(f"No commit SHAs found in {path}")

    if blanks:
        print(f"[WARN] {blanks} blank lines in {path}")

    print(f"[INFO] Loaded {len(commits)} commits from {path}")
    return commits

def _prepare_single_worker(kernel_src: str, tmp_dir: str, idx: int) -> Tuple[int, str]:
    """
    Copy kernel_src → tmp_dir/worker_{idx:03d} if it doesn't already exist.
    Returns (idx, dst_path).
    """
    dst = os.path.join(tmp_dir, f"worker_{idx:03d}")
    if not os.path.exists(dst):
        print(f"[INFO] Copying original kernel to '{dst}'")
        shutil.copytree(kernel_src, dst)
    else:
        print(f"[INFO] Skipping existing worker dir '{dst}'")
    return idx, dst

def prepare_worker_dirs(kernel_src: str,
                        tmp_dir: str,
                        worker_count: int,
                        max_parallelism: int
                        ) -> List[str]:
    """
    Ensure `worker_count` worker directories under `tmp_dir`:
      tmp_dir/worker_000, ..., worker_{worker_count-1:03d}.

    - If a worker directory does not exist: copy kernel_src → that dir.
    - If it already exists: do nothing (we rely on make_patchset to clean).

    Copy tasks are run in parallel, up to `max_parallelism` at once.

    Returns the list of worker directory paths, in index order.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare argument tuples for each worker index
    tasks = [(kernel_src, tmp_dir, i) for i in range(worker_count)]
    results: List[Tuple[int, str]] = []

    # Run copies in parallel, capped at max_parallelism
    with ProcessPoolExecutor(max_workers=max_parallelism) as exe:
        futures = [exe.submit(_prepare_single_worker, *t) for t in tasks]
        for fut in as_completed(futures):
            idx, dst = fut.result()
            results.append((idx, dst))

    # Sort by worker index and return only the paths
    results.sort(key=lambda x: x[0])
    worker_dirs = [dst for _, dst in results]
    return worker_dirs

def make_patchset(
        repo: str,
        idx: int,
        old_sha: str,
        new_sha: str,
        worker_id
) -> Tuple[Path, int, str, str]:  # type: ignore
    """
    Given an `old_sha` and `new_sha`, checkout `new_sha` in `repo`,
    count commits in old..new, and `git diff old..new` → patchset_{idx}.diff.
    Returns (patch_path, commit_count, old_sha, new_sha).
    Retries once if git clean/reset fails.
    """
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ make_patchset: repo={repo}")

    # detect blanks
    if not new_sha:
        raise RuntimeError(f"Blank line in NEW commit list at index {idx}")
    if not old_sha:
        raise RuntimeError(f"Blank line in OLD commit list at index {idx}")

    attempts = 0
    max_attempts = 2
    while attempts < max_attempts:
        try:
            time.sleep(random.uniform(0, 0.5))
            lock_path = Path(repo) / '.git' / 'index.lock'
            if lock_path.exists():
                tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ⚠ Removing stale lock: {lock_path}")
                lock_path.unlink()

            log_path = Path(repo) / f'patchset_{idx}.log'

            # Clean the repo directory
            with open(log_path, "w") as logf:
                subprocess.run(
                    ['git', 'clean', '-dfx'],
                    cwd=repo, check=True,
                    stdout=logf, stderr=subprocess.STDOUT
                )
                subprocess.run(
                    ['git', 'reset', '--hard', 'HEAD'],
                    cwd=repo, check=True,
                    stdout=logf, stderr=subprocess.STDOUT
                )
                subprocess.run(
                    ['git', 'checkout', '-f', new_sha],
                    cwd=repo, check=True,
                    stdout=logf, stderr=subprocess.STDOUT
                )

            break  # success, exit retry loop

        except Exception as e:
            tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] Attempt {attempts+1} failed: {e}")
            if attempts + 1 == max_attempts:
                raise RuntimeError(f"Failed to prepare repo {repo} for patchset {idx} after {max_attempts} attempts") from e
            else:
                tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] Retrying...")
                time.sleep(1)  # optional short backoff
        finally:
            attempts += 1

    # Count commits and generate patch
    cnt = subprocess.check_output(
        ['git', 'rev-list', '--count', '--first-parent', f'{old_sha}..{new_sha}'],
        cwd=repo, text=True
    ).strip()
    commit_count = int(cnt)

    # Generate the patch, log git diff output (stderr) only
    patch_path = Path(repo) / f'patchset_{idx}.diff'
    with open(log_path, "a") as logf, open(patch_path, 'w') as outf:
        subprocess.run(
            ['git', 'diff', f'{old_sha}..{new_sha}'],
            cwd=repo, stdout=outf, stderr=logf, check=True
        )

    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ patchset: {old_sha} → {new_sha} ({commit_count} commits), patch at {patch_path}")

    return patch_path, commit_count, old_sha, new_sha

def run_krepair(repo, patch_path, mode, idx, worker_id):
    """Runs klocalizer in the specified repair mode on the given repo."""
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ run_krepair: mode={mode}, repo={repo}")

    defconfig_log = Path(repo) / 'defconfig_make.log'
    # generate .config
    with open(defconfig_log, "w") as logf:
        subprocess.run(['make', 'defconfig'], cwd=repo, check=True, stdout=logf, stderr=subprocess.STDOUT)

    config_path = Path(repo) / '.config'
    output_file = Path(repo) / f'output_{mode}.txt'
    algo = 'original' if mode == 'krepair' else 'krepairDC'

    # Start building the base command
    cmd = (
        f"source {VENV_ACTIVATE} && klocalizer --repair {config_path} --arch x86_64 "
        f"--include-mutex {patch_path} --mutex-algo {algo} --verbose"
    )

    # Add --num-cores-dc 1 only for krepairDC
    if mode == 'krepairDC':
        cmd += " --num-cores-dc 1"

    # Capture output to file
    with open(output_file, 'w') as outf:
        subprocess.run(
            cmd, cwd=repo, shell=True,
            stdout=outf, stderr=subprocess.STDOUT,
            executable='/bin/bash', check=True
        )

    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ run_krepair done; output saved to {output_file}")
    return output_file

def run_olddefconfig_and_koverage(repo, patch_path, idx, worker_id):
    """Runs olddefconfig and koverage for all *-x86_64.config files in repo."""
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ olddefconfig + koverage on {repo}")
    configs = sorted(str(f) for f in Path(repo).glob('*-x86_64.config'))

    # Run olddefconfig for each config
    for config in configs:
        cmd = f"source {VENV_ACTIVATE} && KCONFIG_CONFIG=\"{config}\" make olddefconfig"
        out_log = config.replace('.config', '_olddefconfig.log')
        with open(Path(repo) / out_log, "w") as logf:
            subprocess.run(cmd, cwd=repo, shell=True, executable='/bin/bash', check=True, stdout=logf, stderr=subprocess.STDOUT)

    # Run koverage for each config
    for config in configs:
        out_json = config.replace('.config', '_coverage_results.json')
        out_log = config.replace('.config', '_koverage.log')
        cmd = (
            f"source {VENV_ACTIVATE} && koverage -f --config \"{config}\" "
            f"--arch x86_64 --check-patch {patch_path} -o \"{out_json}\""
        )
        with open(Path(repo) / out_log, "w") as logf:
            subprocess.run(cmd, cwd=repo, shell=True, executable='/bin/bash', check=True, stdout=logf, stderr=subprocess.STDOUT)

    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ olddefconfig+koverage complete")

def run_defconfig_and_koverage(repo, patch_path, idx, worker_id):
    """
    In defconfig mode, just do `make defconfig` and run koverage once
    on the resulting .config, producing a single JSON.
    """
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ defconfig+koverage on {repo}")
    defconfig_log = Path(repo) / 'defconfig_make.log'
    # regenerate .config
    with open(defconfig_log, "w") as logf:
        subprocess.run(['make', 'defconfig'], cwd=repo, check=True, stdout=logf, stderr=subprocess.STDOUT)

    config_file = Path(repo) / '.config'
    out_json = Path(repo) / 'defconfig_coverage_results.json'
    koverage_log = Path(repo) / 'defconfig_koverage.log'

    # run koverage against the single .config
    cmd = (
        f"source {VENV_ACTIVATE} && "
        f"koverage -f --config {config_file} "
        f"--arch x86_64 --check-patch {patch_path} -o {out_json}"
    )
    with open(koverage_log, "w") as logf:
        subprocess.run(cmd, cwd=repo, shell=True, executable='/bin/bash', check=True, stdout=logf, stderr=subprocess.STDOUT)
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ defconfig+koverage complete")

def compute_patch_coverage(repo, idx, worker_id):
    """
    Runs total_coverage.py on all *_coverage_results.json files in repo,
    then runs patch_coverage.py on the merged output. Returns the coverage ratio.
    Logs stdout/stderr of each step to its own log file, and saves the ratio to the patch coverage log.
    """
    # Get the absolute path to this script's directory
    script_dir = Path(__file__).resolve().parent

    # Build absolute paths to the scripts
    total_coverage_path = (script_dir / '../krepair_evaluation/paper/total_coverage.py').resolve()
    patch_coverage_path = (script_dir / '../krepair_evaluation/paper/patch_coverage.py').resolve()

    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ compute_patch_coverage in {repo}")
    # Collect all coverage result JSON files
    coverage_files = [str(p) for p in Path(repo).glob('*_coverage_results.json')]
    if not coverage_files:
        tqdm.write(f"No coverage result files found in {repo}")
        return None

    total_cov_json = Path(repo) / 'total_coverage_results.json'
    total_cov_log  = Path(repo) / 'total_coverage.log'
    patch_cov_log  = Path(repo) / 'patch_coverage.log'

    # Run total_coverage.py to merge coverage files
    with open(total_cov_log, "w") as logf:
        subprocess.run(
            [
                'python3',
                str(total_coverage_path),
                '-o', str(total_cov_json)
            ] + coverage_files,
            cwd=repo,
            check=True,
            stdout=logf,
            stderr=subprocess.STDOUT
        )

    # Run patch_coverage.py and log output and ratio to file
    patch_proc = subprocess.run(
        [
            'python3',
            str(patch_coverage_path),
            str(total_cov_json)
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    patch_output = patch_proc.stdout.strip()

    # Parse the ratio
    # Parse the *last* number in the line (covered, total, ratio)
    nums = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', patch_output)
    ratio = float(nums[-1]) if nums else None

    # Write everything (output + final ratio) to the log file
    with open(patch_cov_log, "w") as logf:
        print(patch_output, file=logf)
        if ratio is not None:
            print(f"SCRIPT: patch_coverage {ratio}", file=logf)

    if ratio is not None:
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ patch coverage = {ratio}")

    return ratio

def pielou_evenness(values, idx, worker_id) -> Optional[float]:
    """
    Computes Pielou's evenness index (J) for a list of group sizes.
    Returns a value between 0 (completely uneven) and 1 (perfectly even).
    """
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ pielou_evenness on {len(values)} values")
    values = [v for v in values if v > 0]  # Ignore zeroes (as is standard)
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    # Convert to proportions
    proportions = [v / total for v in values]
    # Calculate Shannon entropy H'
    entropy = -sum(p * math.log(p) for p in proportions)
    # Pielou's J: evenness index
    if n == 1:
        return None  # we don't care about evenness of a single group
    pielou_j = entropy / math.log(n)
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ evenness = {pielou_j:.4f}")
    return pielou_j

def parse_summary_csv(summary_file: str, idx, worker_id) -> Tuple[List[int], int, float]:
    """
    Reads the single‐row summary CSV and returns:
      - group_sizes: List[int]
      - total_constraints: int (the CSV's `total_deduped_all` column)
      - time_elapsed_seconds: float (the CSV's `time_elapsed_seconds` column)
    """
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ parse_summary_csv('{summary_file}')")
    with open(summary_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row = next(reader)
        group_sizes = json.loads(row['group_sizes'])
        total_constraints = int(row['total_deduped_all'])
        time_elapsed_seconds = float(row['time_elapsed_seconds'])
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ parsed: groups={len(group_sizes)}, total={total_constraints}, time={time_elapsed_seconds:.2f}s")
        return group_sizes, total_constraints, time_elapsed_seconds

def compute_group_size_range(values: List[int], idx, worker_id) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (largest_group, smallest_group), ignoring zeros.
    If there are no positive values, returns (0, 0).
    """
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ compute_group_size_range on {values}")
    positives = [v for v in values if v > 0]
    if len(positives) < 2:
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] compute_group_size_range: Not enough positive group sizes to compute range.")
        return (None, None)
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ size range = ({max(positives) if positives else 0}, {min(positives) if positives else 0})")
    return (max(positives), min(positives))

def compute_config_change_percentage(
        repo: str,
        original_config: str,
        repaired_configs: List[str],
        idx,
        worker_id
) -> Tuple[float, List[Optional[float]], List[Optional[int]]]:
    """
    Returns:
        mean_pct       – float in [0,1]           (average per-config percentage change)
        per_config_pct – List[Optional[float]]    (percentage change, aligned to repaired_configs;
                                                  None if not a *-x86_64.config or missing data)
        per_config_raw – List[Optional[int]]      (raw change_wrt_original count, same alignment;
                                                  None if not a *-x86_64.config or missing data)
    Also writes these results to 'config_change_percentage.txt'.
    """
    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ▶ compute_config_change_percentage('{original_config}', {len(repaired_configs)} repairs)")

    # get absolute path to script directory and measure_change.py
    script_dir = Path(__file__).resolve().parent
    measure_change_path = (script_dir / '../krepair_evaluation/paper/measure_change.py').resolve()

    # 1. remove old .config if present
    config_path = Path(repo)/'.config'
    if config_path.exists():
        config_path.unlink()
    else:
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] [WARNING] .config did not exist before defconfig regeneration.")

    # 2. regenerate defconfig and log output to cccp_defconfig_make.log
    defconfig_log = Path(repo) / 'cccp_defconfig_make.log'
    with open(defconfig_log, "w") as logf:
        subprocess.run(['make', 'defconfig'], cwd=repo, check=True,
                       stdout=logf, stderr=subprocess.STDOUT)

    # 3. run measure_change.py
    cmd = [
        'python3',
        str(measure_change_path),
        '--original-config', original_config,
        *repaired_configs
    ]
    result = subprocess.run(cmd, cwd=repo,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=True)
    data = json.loads(result.stdout)
    repaired_raw = data.get('repaired', {})
    if isinstance(repaired_raw, list):
        # each entry has a 'configfile' field
        data['repaired'] = {
            entry.get('configfile', f"cfg_{i}"): entry
            for i, entry in enumerate(repaired_raw)
        }

    # 4. sum change_wrt_original (for logging)
    total_changed = sum(
        entry.get('change_wrt_original', 0)
        for entry in data['repaired'].values()
    )

    # 5. count all config options in Kconfig
    kextract_pipeline = (
        "kextract --extract -e ARCH=x86_64 -e SRCARCH=x86 "
        "-e KERNELVERSION=kcu -e srctree=./ -e CC=cc -e LD=ld Kconfig "
        "| grep '^config ' | cut -f2 -d' ' | sort | uniq | wc -l"
    )
    opts_res = subprocess.run(
        kextract_pipeline,
        cwd=repo, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    try:
        total_options = int(opts_res.stdout.strip())
    except ValueError:
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] [WARNING] Unable to parse total option count.")
        # Also write error to file
        with open(os.path.join(repo, "config_change_percentage.txt"), "w") as f:
            f.write("ERROR: Unable to parse total option count.\n")
        return -1.0, [], []

    if total_options <= 0:
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] [WARNING] total_options <= 0, returning -1.")
        with open(os.path.join(repo, "config_change_percentage.txt"), "a") as f:
            f.write("ERROR: total_options <= 0\n")
        return -1.0, [], []

    # 6. build per-config lists aligned with repaired_configs
    per_config_pct: List[Optional[float]] = []
    per_config_raw: List[Optional[int]] = []
    repaired_dict = data.get('repaired', {})

    for cfg in repaired_configs:
        if cfg.endswith("-x86_64.config"):
            key = cfg if cfg in repaired_dict else os.path.basename(cfg)
            entry = repaired_dict.get(key)
            raw = entry.get('change_wrt_original', 0) if entry else None
            pct = (raw / total_options) if raw is not None else None
            per_config_raw.append(raw)
            per_config_pct.append(pct)
        else:
            per_config_raw.append(None)
            per_config_pct.append(None)

    # mean percentage over valid entries
    valid_pcts = [p for p in per_config_pct if p is not None]
    mean_pct = statistics.mean(valid_pcts) if valid_pcts else 0.0

    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] ✓ mean_pct = {mean_pct:.4%}, total_changed = {total_changed}, total_options (in kconfig) = {total_options}, (configs: {len(valid_pcts)})")

    # 7. Write results to config_change_percentage.txt
    with open(os.path.join(repo, "config_change_percentage.txt"), "w") as f:
        f.write(f"Mean percentage change across configs: {mean_pct:.4%}\n\n")
        f.write("Per-config percentage changes:\n")
        for cfg, pct in zip(repaired_configs, per_config_pct):
            if pct is not None:
                f.write(f"  {cfg}: {pct:.4%}\n")
            else:
                f.write(f"  {cfg}: [SKIPPED]\n")

    return mean_pct, per_config_pct, per_config_raw

def gather_patterns(mode: str) -> List[str]:
    """
    Return the list of file patterns to collect for each kernel experiment result,
    depending on the repair mode.

    - For 'defconfig': log, diff, and coverage files only.
    - For 'krepairDC': includes summary, configs, logs, coverage, SMT2 final chunks, etc.
    - For 'krepair'  : includes summary, configs, logs, coverage, patch/original constraints, etc.
    """
    if mode == 'defconfig':
        return [
            'defconfig_koverage.log',
            'defconfig_make.log',
            'patchset_*.diff',
            'total_coverage.log',
            'defconfig_coverage_results.json',
            'patch_coverage.log',
        ]
    elif mode == 'krepairDC':
        return [
            '*-x86_64.config',
            '*_koverage.log',
            '*_coverage_results.json',
            'total_coverage_results.json',
            'defconfig_make.log',
            'cccp_defconfig_make.log',
            'patchset_*.diff',
            'total_coverage.log',
            'patch_coverage.log',
            'config_change_percentage.txt',
            'krepairDC_summary.csv',
            'output_krepairDC.txt',
            'final_chunk_*.smt2',
        ]
    elif mode == 'krepair':
        return [
            '*-x86_64.config',
            '*_koverage.log',
            '*_coverage_results.json',
            'total_coverage_results.json',
            'defconfig_make.log',
            'cccp_defconfig_make.log',
            'patchset_*.diff',
            'total_coverage.log',
            'patch_coverage.log',
            'config_change_percentage.txt',
            'krepair_summary.csv',
            'output_krepair.txt',
            'covered_patch_constraints_*_arch_x86_64.json',
            'patch_constraints.json',
            'original_krepair_smt.smt2',
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def export_job_outputs(
        mode: str,
        repo_dir: str,
        idx: int,
        sha: str,
        export_base: Path
) -> None:
    """
    Copy all experiment result files matching the expected patterns for this mode
    from a worker directory to the export destination for the given job.

    Creates a subfolder: export_base/{idx}_{sha7}/
    and copies each file matching gather_patterns(mode) into it.
    """
    repo_path = Path(repo_dir)
    dest = export_base / f"{idx}_{sha[:7]}"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[EXPORT] Exporting results for job {idx} ({sha[:7]}) from '{repo_dir}' → '{dest}'")

    for pat in gather_patterns(mode):
        matches = list(repo_path.glob(pat))
        print(f"[EXPORT] Pattern '{pat}' → {len(matches)} match(es)")
        if not matches:
            print(f"[EXPORT][WARN] No files matching '{pat}' in {repo_dir}")
        for src in matches:
            try:
                shutil.copy2(src, dest)
                print(f"[EXPORT] Copied '{src.name}' → '{dest}/'")
            except Exception as e:
                print(f"[EXPORT][ERROR] Failed to copy '{src}': {e}", file=sys.stderr)

def process_kernel(args):
    """
    Orchestrates the end-to-end experimental run for a single kernel configuration.
    Depending on mode, runs defconfig, krepair, or krepairDC, collects results,
    and returns a dictionary of experiment metrics.
    """
    worker_id, repo, time_window, idx, mode, old_sha, new_sha = args

    # Initialize all result variables with default values
    code_coverage = 0
    group_sizes = []
    total_constraints = 0
    pielou_j = None
    size_ratio = (None, None)
    time_elapsed_seconds = None
    old_commit: str = ''
    current_commit: str = ''
    commit_count = 0
    config_change_pct = None
    per_config_pct = []
    per_config_change = []

    tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] Started processing kernel {repo}")

    try:
        # 1. Generate the patch between historical and selected commit.
        #    Also get the number of commits in the diff range.
        patch, commit_count, old_commit, current_commit = make_patchset(
            repo, idx, old_sha, new_sha, worker_id
        )

        # 2. Run the appropriate experiment step
        if mode == 'defconfig':
            # Run defconfig and coverage analysis
            run_defconfig_and_koverage(repo, patch, idx, worker_id)
        else:
            # Run krepair/krepairDC and follow-up coverage analysis
            run_krepair(repo, patch, mode, idx, worker_id)
            run_olddefconfig_and_koverage(repo, patch, idx, worker_id)

        # 3. Collect code coverage results from coverage tool
        code_coverage = compute_patch_coverage(repo, idx, worker_id)

        # 4. For krepair modes, gather group stats and timing from summary CSV
        if mode != 'defconfig':
            summary_file = str(Path(repo) / ('krepair_summary.csv' if mode=='krepair' else 'krepairDC_summary.csv'))
            # Parse group sizes, total constraint count, and elapsed time
            group_sizes, total_constraints, time_elapsed_seconds = parse_summary_csv(summary_file, idx, worker_id)
            # Compute Pielou's evenness and group size ratio
            pielou_j   = pielou_evenness(group_sizes, idx, worker_id)
            size_ratio = compute_group_size_range(group_sizes, idx, worker_id)

            # measure how much configs changed under krepair/krepairDC
            repaired_configs = sorted(
                    str(p) for p in Path(repo).glob('*-x86_64.config')
                )
            config_change_pct, per_config_pct, per_config_change = compute_config_change_percentage(
                    repo,
                    str(Path(repo) / '.config'),
                    repaired_configs,
                    idx,
                    worker_id
                )

        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] Finished processing kernel {repo}")

        # 5. Return all experiment results as a dictionary
        return {
            'job_index':            idx,
            'mode':                 mode,
            'time_window':          time_window,
            'old_commit':           old_commit,
            'current_commit':       current_commit,
            'commit_count':         commit_count,
            'coverage':             code_coverage,
            'groups':               group_sizes,
            'num_groups':           len(group_sizes),
            'total_constraints':    total_constraints,
            'evenness':             pielou_j,
            'size_ratio':           size_ratio,
            'time_elapsed_seconds': time_elapsed_seconds,
            'config_change_pct':    config_change_pct,
            'per_config_pct':       per_config_pct,
            'per_config_change':    per_config_change
        }

    except Exception as e:
        tqdm.write(f"[JOB {idx}] [WORKER {worker_id}] Error processing kernel {repo}: {e}")
        return {
            'job_index': idx,
            'mode':      mode,
            'old_commit': old_commit,
            'current_commit': current_commit,
            'commit_count': commit_count,
            'time_window': time_window,
            'skip_reason': str(e)[:350]      # truncate long tracebacks
        }

def write_results_to_csv(results, csv_path):
    """
    Writes a list of result dictionaries to a CSV file.
    Flattens the 'groups' list as JSON and 'size_ratio' tuple as "max,min".
    Ensures every fieldname is present (filling missing ones with '' or [] as appropriate).
    Skips any None entries.
    """
    fieldnames = [
        'job_index',
        'mode',
        'time_window',
        'old_commit',
        'current_commit',
        'commit_count',
        'coverage',
        'groups',
        'num_groups',
        'total_constraints',
        'evenness',
        'size_ratio',
        'time_elapsed_seconds',
        'config_change_pct',
        'per_config_pct',
        'per_config_change',
        'skip_reason'
    ]

    # Make sure target dir exists
    out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Writing {len(results)} result rows to '{csv_path}'")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # type: ignore
        writer.writeheader()

        for result in results:
            if not result:
                continue

            # Copy so we don’t mutate the original
            row = result.copy()

            # Ensure every column has *something*
            for key in fieldnames:
                if key in ('per_config_pct', 'per_config_change', 'groups'):
                    row.setdefault(key, [])
                else:
                    row.setdefault(key, '')

            # Flatten the complex fields
            row['groups'] = json.dumps(row['groups'])
            if row['size_ratio'] and row['size_ratio'][0] is not None:
                row['size_ratio'] = f"{row['size_ratio'][0]},{row['size_ratio'][1]}"
            else:
                row['size_ratio'] = ''

            # config_change_pct and evenness can be None or a float
            if row['config_change_pct'] is None:
                row['config_change_pct'] = ''
            if row['evenness'] is None:
                row['evenness'] = ''

            # per_config_pct and per_config_change are lists --> JSON-encode
            row['per_config_pct'] = json.dumps(row['per_config_pct'])
            row['per_config_change'] = json.dumps(row['per_config_change'])

            # skip_reason stays as-is ('' if not present)
            writer.writerow(row)


def main():
    """Parses arguments and dispatches the parallel kernel processing."""
    parser = argparse.ArgumentParser(
        description='Run krepair and krepairDC on provided kernel commits.'
    )
    parser.add_argument(
        '--num-kernels', '-n', type=int, default=385,
        help='Maximum number of kernels (i.e. commits) to process (default: 385).'
    )
    parser.add_argument(
        '--kernels-src', type=str, required=True,
        help='Path to the directory containing source kernel repos.'
    )
    parser.add_argument(
        '--tmp-dir', type=str, default='/tmp/testing_kernels',
        help='Path for temporary working directory (default: /tmp/testing_kernels).'
    )
    parser.add_argument(
        '--cores', type=int, default=48,
        help='Number of parallel jobs (default: 48).'
    )
    parser.add_argument(
        '--commit-list-file', type=str, required=True,
        help='Path to file listing new (selected) commits.'
    )
    parser.add_argument(
        '--old-commit-list-file', type=str, required=True,
        help='Path to file listing old (reference) commits.'
    )
    parser.add_argument(
        'time_window', type=str,
        help='Time window label for patchset (e.g., 12h, 72h, 7d). Only used for labeling in CSV output.'
    )
    parser.add_argument(
        '--mode', choices=['krepair', 'krepairDC', 'defconfig'], default='krepair',
        help='Repair mode: krepair for krepair, krepairDC for krepairDC, defconfig for non-repaired defconfig .config coverage'
    )
    parser.add_argument(
        '--export-dir', type=str, default=None, required=True,
        help='Path to write the per-run exports, including the final CSV file.'
    )
    args = parser.parse_args()

    # Assign args to variables/global config as needed
    kernels_src = args.kernels_src
    tmp_dir = args.tmp_dir
    cores = args.cores
    commit_list_file = args.commit_list_file
    old_commit_list_file = args.old_commit_list_file
    time_window = args.time_window
    mode = args.mode
    num_kernels = args.num_kernels
    export_root = Path(args.export_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_base = export_root / f"{mode}_{timestamp}"
    export_base.mkdir(parents=True, exist_ok=True)

    # decide where to write our aggregate CSV: place it inside export_base
    output_csv = export_base / f"results_{mode}_{time_window}.csv"

    # Make sure the temporary directory exists
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare worker dirs
    worker_dirs = prepare_worker_dirs(
        kernels_src,
        tmp_dir,
        worker_count=cores,
        max_parallelism=cores
    )

    # Load & split commits
    commits     = load_commits(commit_list_file)[:num_kernels]
    old_commits = load_commits(old_commit_list_file)[:num_kernels]

    # Build pending-commit list (idx, old_sha, new_sha)
    pending = deque(
        (idx, old_sha, new_sha)
        for idx, (new_sha, old_sha) in enumerate(zip(commits, old_commits))
        if new_sha and old_sha
    )
    skipped = [idx for idx, (n,o) in enumerate(zip(commits, old_commits)) if not n or not o]

    # Pre-populate blank-line skips in results
    results: List[Dict[str, Any]] = [
        {
            'job_index':            idx,
            'mode':                 mode,
            'time_window':          time_window,
            'old_commit':           '',
            'current_commit':       '',
            'coverage':             None,
            'groups':               [],
            'num_groups':           0,
            'total_constraints':    0,
            'evenness':             None,
            'size_ratio':           (None, None),
            'time_elapsed_seconds': None,
            'commit_count':         None,
            'config_change_pct':    None,
            'per_config_pct':       [],
            'per_config_change':    [],
            'skip_reason':          'blank line in commit list'
        }
        for idx in skipped
    ]

    total_jobs = len(pending)                         # all non-blank commits
    available_workers = deque(range(cores))           # 0 … cores-1
    futures: Dict[Any, Tuple[int, int, str]] = {}     # Future → (wid, idx, sha)

    with ProcessPoolExecutor(max_workers=cores) as exe:
        # Kick off the first batch (≤ cores jobs)
        while available_workers and pending:
            wid = available_workers.popleft()
            idx, old_sha, new_sha = pending.popleft()
            repo = worker_dirs[wid]
            fut = exe.submit(
                process_kernel,
                (wid, repo, time_window, idx, mode, old_sha, new_sha)
            )
            futures[fut] = (wid, idx, new_sha)

        with tqdm(total=total_jobs, desc=f"[{mode}] jobs", unit="job") as pbar:
            while futures:
                # Wait for any currently-running job to finish
                done = next(as_completed(futures), None)
                wid, idx, sha = futures.pop(done)          # remove from map
                res = done.result()
                results.append(res)
                pbar.update(1)

                # Export immediately if it ran at all
                export_job_outputs(mode, worker_dirs[wid], idx, sha, export_base)

                # Mark this worker as free
                available_workers.append(wid)

                # Launch one new job if commits remain
                if pending and available_workers:
                    wid2 = available_workers.popleft()
                    idx2, old2, new2 = pending.popleft()
                    repo2 = worker_dirs[wid2]
                    fut2 = exe.submit(
                        process_kernel,
                        (wid2, repo2, time_window, idx2, mode, old2, new2)
                    )
                    futures[fut2] = (wid2, idx2, new2)

    # When done, write CSV
    results.sort(key=lambda r: r['job_index'])
    write_results_to_csv(results, output_csv)

    # Count only truly successful runs (no skip_reason at all)
    success_count = sum(1 for r in results if not r.get('skip_reason'))
    total = len(results)
    failed = total - success_count

    print(f"Done ({time_window}, {mode}): {success_count} succeeded, {failed} failed or skipped out of {total}.")

if __name__ == '__main__':
    main()