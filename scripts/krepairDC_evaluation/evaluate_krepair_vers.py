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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

def copy_kernel_multiple_times(kernel_src: str,
                               tmp_dir: str,
                               mode: str,
                               old_commit_list_file: str,
                               commit_list_file: str,
                               max_kernels: int,
                               cores: int
                               ) -> Tuple[List[Tuple[int, str, str, str]], List[int]]:
    """
    Parallel-copy the kernel tree once per *non-blank* commit, placing each copy
    under tmp_dir/<mode>/, using up to `cores` workers.
    Returns:
      copied   – [(idx, dst_path), …]  sorted by idx
      skipped  – [idx, idx, …]         for blank-line entries
    """
    print(f"[INFO] Copying kernels from {kernel_src} to {tmp_dir}/{mode} (max {max_kernels} kernels, {cores} cores)")
    dest_root = os.path.join(tmp_dir, mode)
    os.makedirs(dest_root, exist_ok=True)

    commits = load_commits(commit_list_file)[:max_kernels]
    old_commits = load_commits(old_commit_list_file)[:max_kernels]

    tasks: List[Tuple[int,str,str,str]] = []
    skipped: List[int] = []
    for idx, (new_sha, old_sha) in enumerate(zip(commits, old_commits)):
        if not new_sha or not old_sha:
            skipped.append(idx)
        else:
            dst = os.path.join(dest_root, f"{idx:03d}_{new_sha[:7]}")
            tasks.append((idx, new_sha, old_sha, dst))

    results: List[Tuple[int, str, str, str]] = []
    with ProcessPoolExecutor(max_workers=cores) as execr:
        # pass kernel_src and each task to the top-level function
        future_to_idx = {
            execr.submit(_copy_single_kernel_task, kernel_src, t): t[0]
            for t in tasks
        }
        for fut in as_completed(future_to_idx):
            results.append(fut.result())

    results.sort(key=lambda x: x[0])
    return results, skipped

def _copy_single_kernel_task(kernel_src: str,
                             task: Tuple[int, str, str, str]
                             ) -> Tuple[int, str, str, str]:
    idx, new_sha, old_sha, dst_path = task
    """
    Top-level helper so it can be pickled.
    task = (idx, sha, dst_path)
    """
    print(f"[INFO] Copying kernel #{idx} to '{dst_path}'")
    if not os.path.exists(dst_path):
        shutil.copytree(kernel_src, dst_path)
    return idx, dst_path, old_sha, new_sha

def make_patchset(
        repo: str,
        idx: int,
        old_sha: str,
        new_sha: str
) -> Tuple[Path, int, str, str]:
    """
    Given an `old_sha` and `new_sha`, checkout `new_sha` in `repo`,
    count commits in old..new, and `git diff old..new` → patchset_{idx}.diff.
    Returns (patch_path, commit_count, old_sha, new_sha).
    """
    tqdm.write(f"[JOB {idx}] ▶ make_patchset: repo={repo}")

    # detect blanks
    if not new_sha:
        raise RuntimeError(f"Blank line in NEW commit list at index {idx}")
    if not old_sha:
        raise RuntimeError(f"Blank line in OLD commit list at index {idx}")

    log_path = Path(repo) / f'patchset_{idx}.log'

    # Clean the repo directory
    with open(log_path, "w") as logf:
        subprocess.run(
            ['git', 'clean', '-dfx'],
            cwd=repo, check=True,
            stdout=logf, stderr=subprocess.STDOUT
        )

        # Checkout the new commit
        subprocess.run(
            ['git', 'checkout', '-f', new_sha],
            cwd=repo, check=True,
            stdout=logf, stderr=subprocess.STDOUT
        )

    # Count commits in old_sha..new_sha (no need to log; returns directly)
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

    tqdm.write(f"[JOB {idx}] ✓ patchset: {old_sha} → {new_sha} ({commit_count} commits), patch at {patch_path}")

    return patch_path, commit_count, old_sha, new_sha

def run_krepair(repo, patch_path, mode, idx):
    """Runs klocalizer in the specified repair mode on the given repo."""
    tqdm.write(f"[JOB {idx}] ▶ run_krepair: mode={mode}, repo={repo}")

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

    tqdm.write(f"[JOB {idx}] ✓ run_krepair done; output saved to {output_file}")
    return output_file

def run_olddefconfig_and_koverage(repo, patch_path, idx):
    """Runs olddefconfig and koverage for all *-x86_64.config files in repo."""
    tqdm.write(f"[JOB {idx}] ▶ olddefconfig + koverage on {repo}")
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

    tqdm.write(f"[JOB {idx}] ✓ olddefconfig+koverage complete")

def run_defconfig_and_koverage(repo, patch_path, idx):
    """
    In defconfig mode, just do `make defconfig` and run koverage once
    on the resulting .config, producing a single JSON.
    """
    tqdm.write(f"[JOB {idx}] ▶ defconfig+koverage on {repo}")
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
    tqdm.write(f"[JOB {idx}] ✓ defconfig+koverage complete")

def compute_patch_coverage(repo, idx):
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

    tqdm.write(f"[JOB {idx}] ▶ compute_patch_coverage in {repo}")
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
        tqdm.write(f"[JOB {idx}] ✓ patch coverage = {ratio}")

    return ratio

def pielou_evenness(values, idx) -> Optional[float]:
    """
    Computes Pielou's evenness index (J) for a list of group sizes.
    Returns a value between 0 (completely uneven) and 1 (perfectly even).
    """
    tqdm.write(f"[JOB {idx}] ▶ pielou_evenness on {len(values)} values")
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
    tqdm.write(f"[JOB {idx}] ✓ evenness = {pielou_j:.4f}")
    return pielou_j

def parse_summary_csv(summary_file: str, idx) -> Tuple[List[int], int, float]:
    """
    Reads the single‐row summary CSV and returns:
      - group_sizes: List[int]
      - total_constraints: int (the CSV's `total_deduped_all` column)
      - time_elapsed_seconds: float (the CSV's `time_elapsed_seconds` column)
    """
    tqdm.write(f"[JOB {idx}] ▶ parse_summary_csv('{summary_file}')")
    with open(summary_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row = next(reader)
        group_sizes = json.loads(row['group_sizes'])
        total_constraints = int(row['total_deduped_all'])
        time_elapsed_seconds = float(row['time_elapsed_seconds'])
        tqdm.write(f"[JOB {idx}] ✓ parsed: groups={len(group_sizes)}, total={total_constraints}, time={time_elapsed_seconds:.2f}s")
        return group_sizes, total_constraints, time_elapsed_seconds

def compute_group_size_range(values: List[int], idx) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (largest_group, smallest_group), ignoring zeros.
    If there are no positive values, returns (0, 0).
    """
    tqdm.write(f"[JOB {idx}] ▶ compute_group_size_range on {values}")
    positives = [v for v in values if v > 0]
    if len(positives) < 2:
        tqdm.write(f"[JOB {idx}] compute_group_size_range: Not enough positive group sizes to compute range.")
        return (None, None)
    tqdm.write(f"[JOB {idx}] ✓ size range = ({max(positives) if positives else 0}, {min(positives) if positives else 0})")
    return (max(positives), min(positives))

def compute_config_change_percentage(
        repo: str,
        original_config: str,
        repaired_configs: List[str],
        idx
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
    tqdm.write(f"[JOB {idx}] ▶ compute_config_change_percentage('{original_config}', {len(repaired_configs)} repairs)")

    # get absolute path to script directory and measure_change.py
    script_dir = Path(__file__).resolve().parent
    measure_change_path = (script_dir / '../krepair_evaluation/paper/measure_change.py').resolve()

    # 1. remove old .config if present
    config_path = Path(repo)/'.config'
    if config_path.exists():
        config_path.unlink()
    else:
        tqdm.write("[WARNING] .config did not exist before defconfig regeneration.")

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
        tqdm.write("[WARNING] Unable to parse total option count.")
        # Also write error to file
        with open(os.path.join(repo, "config_change_percentage.txt"), "w") as f:
            f.write("ERROR: Unable to parse total option count.\n")
        return -1.0, [], []

    if total_options <= 0:
        tqdm.write("[WARNING] total_options <= 0, returning -1.")
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

    tqdm.write(f"[JOB {idx}] ✓ mean_pct = {mean_pct:.4%}, total_changed = {total_changed}, total_options (in kconfig) = {total_options}, (configs: {len(valid_pcts)})")

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

def process_kernel(args):
    """
    Orchestrates the end-to-end experimental run for a single kernel configuration.
    Depending on mode, runs defconfig, krepair, or krepairDC, collects results,
    and returns a dictionary of experiment metrics.
    """
    repo, time_window, idx, mode, old_sha, new_sha = args

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
    per_config_raw = []

    tqdm.write(f"[JOB {idx}] Started processing kernel {repo}")

    try:
        # 1. Generate the patch between historical and selected commit.
        #    Also get the number of commits in the diff range.
        patch, commit_count, old_commit, current_commit = make_patchset(
            repo, idx, old_sha, new_sha
        )

        # 2. Run the appropriate experiment step
        if mode == 'defconfig':
            # Run defconfig and coverage analysis
            run_defconfig_and_koverage(repo, patch, idx)
        else:
            # Run krepair/krepairDC and follow-up coverage analysis
            run_krepair(repo, patch, mode, idx)
            run_olddefconfig_and_koverage(repo, patch, idx)

        # 3. Collect code coverage results from coverage tool
        code_coverage = compute_patch_coverage(repo, idx)

        # 4. For krepair modes, gather group stats and timing from summary CSV
        if mode != 'defconfig':
            summary_file = str(Path(repo) / ('krepair_summary.csv' if mode=='krepair' else 'krepairDC_summary.csv'))
            # Parse group sizes, total constraint count, and elapsed time
            group_sizes, total_constraints, time_elapsed_seconds = parse_summary_csv(summary_file, idx)
            # Compute Pielou's evenness and group size ratio
            pielou_j   = pielou_evenness(group_sizes, idx)
            size_ratio = compute_group_size_range(group_sizes, idx)

            # measure how much configs changed under krepair/krepairDC
            repaired_configs = sorted(
                    str(p) for p in Path(repo).glob('*-x86_64.config')
                )
            config_change_pct, per_config_pct, per_config_raw = compute_config_change_percentage(
                    repo,
                    str(Path(repo) / '.config'),
                    repaired_configs,
                    idx
                )

        tqdm.write(f"[JOB {idx}] Finished processing kernel {repo}")

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
            'per_config_raw':       per_config_raw
        }

    except Exception as e:
        tqdm.write(f"[JOB {idx}] Error processing kernel {repo}: {e}")
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
        'per_config_raw',
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
                if key in ('per_config_pct', 'per_config_raw', 'groups'):
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

            # per_config_pct and per_config_raw are lists --> JSON-encode
            row['per_config_pct'] = json.dumps(row['per_config_pct'])
            row['per_config_raw'] = json.dumps(row['per_config_raw'])

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
        '--output-csv', type=str, default=None,
        help='Path to write the aggregated results CSV. '
             'Defaults to ./results_{mode}_{time_window}.csv'
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

    # Set output CSV default dynamically if not specified
    if args.output_csv is None:
        output_csv = f'results_{mode}_{time_window}.csv'
    else:
        output_csv = args.output_csv

    # Make sure the temporary directory exists
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare kernel repos
    copy_cores = min(cores, 8)  # limit copy parallelism to avoid overload
    copied_kernels, skipped_idxs = copy_kernel_multiple_times(
        kernels_src, tmp_dir, mode, old_commit_list_file, commit_list_file, num_kernels, copy_cores
    )

    # Pre-create result rows for every blank-line skip
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
            'per_config_raw':       [],
            'skip_reason':          'blank line in commit list'
        }
        for idx in skipped_idxs
    ]

    # Build a list of jobs: (kernel_dir, time_window, job_index, mode, old_sha, new_sha) for each kernel
    jobs = [
        (kernel_dir, time_window, commit_idx, mode, old_sha, new_sha)
        for (commit_idx, kernel_dir, old_sha, new_sha) in copied_kernels
    ]

    print(f"[INFO] Scheduling {len(jobs)} jobs in mode='{mode}'")
    for commit_idx, kernel_dir, old_sha, new_sha in copied_kernels:
        print(f"[INFO]  • Job {commit_idx}: kernel dir = {kernel_dir}")

    # Process the jobs in parallel
    with ProcessPoolExecutor(max_workers=cores) as executor:
        # 1) submit all the jobs
        futures = [executor.submit(process_kernel, job) for job in jobs]

        # 2) create a tqdm bar up front
        with tqdm(total=len(futures),
                  desc=f"[{mode}] jobs",
                  unit="job") as pbar:
            # 3) as each future completes, grab it and advance the bar
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    results.sort(key=lambda r: r['job_index'])  # just in case

    # Write the results to CSV
    write_results_to_csv(results, output_csv)

    # Output the result summary
    success_count = sum(1 for r in results if not r.get('skip_reason'))
    print(f"Done ({args.time_window}, {mode}). {success_count} runs succeeded.")

if __name__ == '__main__':
    main()