import re
import os
import csv
import math
import json
import shutil
import subprocess
import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

# Constants for number of kernels and paths
VENV_ACTIVATE = os.path.expanduser('/home/alexei/IDEProjects/PyCharmProjects/kmax/venv_new/bin/activate')
KOVERAGE = 'koverage'  # Assumes koverage is in the virtualenv's path

# Time modes
TIME_WINDOWS = ['12h', '72h', '7d']

def load_commits(path: str) -> List[str]:
    with open(path) as f:
        commits = [ln.strip() for ln in f if ln.strip()]
    # oldest-to-newest for repeatability
    commits.reverse()
    print(f"[INFO] Loaded {len(commits)} commits from {path}")
    return commits

def copy_kernel_multiple_times(kernel_src: str,
                               tmp_dir: str,
                               commits: List[str]
                               ) -> List[Tuple[int, str]]:
    """
    One copy per commit.  Returns [(commit_idx, dst_path), …]
    """
    copied = []
    for idx, sha in enumerate(commits):
        dst_path = os.path.join(tmp_dir, f"{idx:03d}_{sha[:7]}")
        print(f"[INFO] Copying kernel #{idx} to '{dst_path}'")
        if not os.path.exists(dst_path):
            shutil.copytree(kernel_src, dst_path)
        copied.append((idx, dst_path))
    return copied

def make_patchset(
        repo: str,
        idx: int,
        commit_list_file: str,
        old_commit_list_file: str
) -> Tuple[Path, int, str, str]:
    """
    1) Load 'new' commits from commit_list_file
    2) Load 'old' commits from old_commit_list_file
    3) Pick both by the same index (wrapping around)
    4) Checkout the 'new' commit
    5) Count how many commits are between old..new
    6) Generate a diff patch at repo/patchset_{idx}.diff
    Returns: (patch_path, commit_count)
    """
    print(f"[JOB {idx}] ▶ make_patchset: repo={repo}")
    # — Load lists of SHAs
    with open(commit_list_file, 'r') as f:
        new_commits = [c.strip() for c in f if c.strip()]
    if not new_commits:
        raise RuntimeError(f"No commits in {commit_list_file}")

    with open(old_commit_list_file, 'r') as f:
        old_commits = [c.strip() for c in f if c.strip()]
    if not old_commits:
        raise RuntimeError(f"No commits in {old_commit_list_file}")

    # — Select by index (wrap if idx >= len)
    selected = new_commits[idx % len(new_commits)]
    reference = old_commits[idx % len(old_commits)]

    # Bail out if the chosen old-commit entry is empty
    if not reference:
        raise RuntimeError(
            f"Empty old-commit entry at line {idx} of {old_commit_list_file}"
        )

    # — Checkout the new commit
    subprocess.run(
        ['git', 'checkout', '-f', selected],
        cwd=repo, check=True
    )

    # — Count how many commits are in the range old..new
    cnt = subprocess.check_output(
        ['git', 'rev-list', '--count', f'{reference}..{selected}'],
        cwd=repo,
        text=True
    ).strip()
    commit_count = int(cnt)

    # — Generate the patch
    patch_path = Path(repo) / f'patchset_{idx}.diff'
    with open(patch_path, 'w') as outf:
        subprocess.run(
            ['git', 'diff', f'{reference}..{selected}'],
            cwd=repo, stdout=outf, check=True
        )

    print(f"[JOB {idx}] ✓ patchset: {reference} → {selected} ({commit_count} commits), patch at {patch_path}")

    return patch_path, commit_count, reference, selected

def run_krepair(repo, patch_path, mode, idx):
    """Runs klocalizer in the specified repair mode on the given repo."""
    print(f"[JOB {idx}] ▶ run_krepair: mode={mode}, repo={repo}")

    subprocess.run(['make', 'defconfig'], cwd=repo, check=True)
    config_path = Path(repo) / '.config'
    output_file = Path(repo) / f'output_{mode}.txt'
    algo = 'original' if mode == 'original' else 'krepairDC'

    cmd = (
        f"source {VENV_ACTIVATE} && klocalizer --repair {config_path} --arch x86_64 "
        f"--include-mutex {patch_path} --mutex-algo {algo} --verbose"
    )

    # Capture output to file
    with open(output_file, 'w') as outf:
        subprocess.run(
            cmd, cwd=repo, shell=True,
            stdout=outf, stderr=subprocess.STDOUT,
            executable='/bin/bash', check=True
        )

    print(f"[JOB {idx}] ✓ run_krepair done; output saved to {output_file}")
    return output_file

def run_olddefconfig_and_koverage(repo, patch_path, idx):
    """Runs olddefconfig and koverage for all *-x86_64.config files in repo."""
    print(f"[JOB {idx}] ▶ olddefconfig + koverage on {repo}")
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
            f"source {VENV_ACTIVATE} && {KOVERAGE} -f --config \"{config}\" "
            f"--arch x86_64 --check-patch {patch_path} -o \"{out_json}\""
        )
        with open(Path(repo) / out_log, "w") as logf:
            subprocess.run(cmd, cwd=repo, shell=True, executable='/bin/bash', check=True, stdout=logf, stderr=subprocess.STDOUT)

    print(f"[JOB {idx}] ✓ olddefconfig+koverage complete")

def run_defconfig_and_koverage(repo, patch_path, idx):
    """
    In defconfig mode, just do `make defconfig` and run koverage once
    on the resulting .config, producing a single JSON.
    """
    print(f"[JOB {idx}] ▶ defconfig+koverage on {repo}")
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
        f"{KOVERAGE} -f --config {config_file} "
        f"--arch x86_64 --check-patch {patch_path} -o {out_json}"
    )
    with open(koverage_log, "w") as logf:
        subprocess.run(cmd, cwd=repo, shell=True, executable='/bin/bash', check=True, stdout=logf, stderr=subprocess.STDOUT)
    print(f"[JOB {idx}] ✓ defconfig+koverage complete")

def compute_patch_coverage(repo, idx):
    """
    Runs total_coverage.py on all *_coverage_results.json files in repo,
    then runs patch_coverage.py on the merged output. Returns the coverage ratio.
    Logs stdout/stderr of each step to its own log file, and saves the ratio to the patch coverage log.
    """
    print(f"[JOB {idx}] ▶ compute_patch_coverage in {repo}")
    # Collect all coverage result JSON files
    coverage_files = [str(p) for p in Path(repo).glob('*_coverage_results.json')]
    if not coverage_files:
        print(f"No coverage result files found in {repo}")
        return None

    total_cov_path = Path(repo) / 'total_coverage_results.json'
    total_cov_log  = Path(repo) / 'total_coverage.log'
    patch_cov_log  = Path(repo) / 'patch_coverage.log'

    # Run total_coverage.py to merge coverage files
    with open(total_cov_log, "w") as logf:
        subprocess.run(
            [
                'python3',
                '../krepair_evaluation/paper/total_coverage.py',
                '-o', str(total_cov_path)
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
            '../krepair_evaluation/paper/patch_coverage.py',
            str(total_cov_path)
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
            print(f"patch_coverage_ratio {ratio}", file=logf)

    if ratio is not None:
        print(f"[JOB {idx}] ✓ patch coverage ratio = {ratio}")

    return ratio

def pielou_evenness(values, idx):
    """
    Computes Pielou's evenness index (J) for a list of group sizes.
    Returns a value between 0 (completely uneven) and 1 (perfectly even).
    """
    print(f"[JOB {idx}] ▶ pielou_evenness on {len(values)} values")
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
        return 1.0  # Perfectly even by definition (only one group)
    pielou_j = entropy / math.log(n)
    print(f"[JOB {idx}] ✓ evenness = {pielou_j:.4f}")
    return pielou_j

def parse_summary_csv(summary_file: str, idx) -> Tuple[List[int], int, float]:
    """
    Reads the single‐row summary CSV and returns:
      - group_sizes: List[int]
      - total_constraints: int (the CSV's `total_deduped_all` column)
      - time_elapsed_seconds: float (the CSV's `time_elapsed_seconds` column)
    """
    print(f"[JOB {idx}] ▶ parse_summary_csv('{summary_file}')")
    with open(summary_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row = next(reader)
        group_sizes = json.loads(row['group_sizes'])
        total_constraints = int(row['total_deduped_all'])
        time_elapsed_seconds = float(row['time_elapsed_seconds'])
        print(f"[JOB {idx}] ✓ parsed: groups={len(group_sizes)}, total={total_constraints}, time={time_elapsed_seconds:.2f}s")
        return group_sizes, total_constraints, time_elapsed_seconds

def compute_group_size_range(values: List[int], idx) -> Tuple[int, int]:
    """
    Returns (largest_group, smallest_group), ignoring zeros.
    If there are no positive values, returns (0, 0).
    """
    print(f"[JOB {idx}] ▶ compute_group_size_range on {values}")
    positives = [v for v in values if v > 0]
    if not positives:
        return (0, 0)
    print(f"[JOB {idx}] ✓ size range = ({max(positives) if positives else 0}, {min(positives) if positives else 0})")
    return (max(positives), min(positives))

def compute_config_change_percentage(
        repo: str,
        original_config: str,
        repaired_configs: List[str],
        idx
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
        mean_pct   – float   in [0,1]  (average per-config percentage change)
        per_config – dict[str,float]   {cfg → pct_i}
    Also writes these results to 'config_change_percentage.txt'.
    """
    print(f"[JOB {idx}] ▶ compute_config_change_percentage('{original_config}', {len(repaired_configs)} repairs)")

    # 1. remove old .config if present
    config_path = Path(repo)/'.config'
    if config_path.exists():
        config_path.unlink()
    else:
        print("[WARNING] .config did not exist before defconfig regeneration.")

    # 2. regenerate defconfig
    subprocess.run(['make', 'defconfig'], cwd=repo, check=True)

    # 3. measure_change.py (hard-coded path)
    cmd = [
        'python3',
        '../krepair_evaluation/paper/measure_change.py',
        '--original-config', original_config,
        *repaired_configs
    ]
    result = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=True)
    data = json.loads(result.stdout)

    # 4. sum change_wrt_original (kept for logging / compatibility)
    total_changed = sum(
        entry.get('change_wrt_original', 0)
        for entry in data.get('repaired', {}).values()
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
        print("[WARNING] Unable to parse total option count.")
        # Also write error to file
        with open(os.path.join(repo, "config_change_percentage.txt"), "w") as f:
            f.write("ERROR: Unable to parse total option count.\n")
        return -1.0, {}

    if total_options <= 0:
        print("[WARNING] total_options <= 0, returning -1.")
        with open(os.path.join(repo, "config_change_percentage.txt"), "a") as f:
            f.write("ERROR: total_options <= 0\n")
        return -1.0, {}

    # 6. build per-config percentage dict
    per_config_pct = {
        cfg: info.get('change_wrt_original', 0) / total_options
        for cfg, info in data.get('repaired', {}).items()
    }

    mean_pct = statistics.mean(per_config_pct.values()) if per_config_pct else 0.0

    print(f"[JOB {idx}] ✓ mean_pct = {mean_pct:.4%}, total_changed = {total_changed:.4%}, (configs: {len(per_config_pct)})")

    # 7. Write results to config_change_percentage.txt
    with open(os.path.join(repo, "config_change_percentage.txt"), "w") as f:
        f.write(f"Mean percentage change across configs: {mean_pct:.4%}\n\n")
        f.write("Per-config percentage changes:\n")
        for cfg, pct in per_config_pct.items():
            f.write(f"  {cfg}: {pct:.4%}\n")

    return mean_pct, per_config_pct

def process_kernel(args):
    """
    Orchestrates the end-to-end experimental run for a single kernel configuration.
    Depending on mode, runs defconfig, krepair, or krepairDC, collects results,
    and returns a dictionary of experiment metrics.
    """
    repo, time_window, idx, mode, commit_list_file, old_commit_list_file = args

    # Initialize all result variables with default values
    code_coverage = 0
    group_sizes = []
    total_constraints = 0
    pielou_j = 0
    size_ratio = (None, None)
    time_elapsed_seconds = None
    old_commit: str = ''
    current_commit: str = ''
    config_change_pct = None
    per_config_pct = {}

    try:
        # 1. Generate the patch between historical and selected commit.
        #    Also get the number of commits in the diff range.
        patch, commit_count, old_commit, current_commit = make_patchset(
            repo, idx, commit_list_file, old_commit_list_file
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
                    str(p) for p in Path(repo).glob('*-repaired.config')
                )
            config_change_pct, per_config_pct = compute_config_change_percentage(
                    repo,
                    str(Path(repo) / '.config'),
                    repaired_configs,
                    idx
                )

        # 5. Return all experiment results as a dictionary
        return {
            'job_index':            idx,
            'mode':                 mode,
            'time_window':          time_window,
            'old_commit':           old_commit,
            'current_commit':       current_commit,
            'coverage':             code_coverage,
            'groups':               group_sizes,
            'total_constraints':    total_constraints,
            'evenness':             pielou_j,
            'size_ratio':           size_ratio,
            'time_elapsed_seconds': time_elapsed_seconds,
            'commit_count':         commit_count,
            'config_change_pct':    config_change_pct,
            'per_config_pct':       per_config_pct
        }

    except Exception as e:
        print(f"Error in {repo}: {e}")
        return None

def write_results_to_csv(results, csv_path):
    """
    Writes a list of result dictionaries to a CSV file.
    Flattens the 'groups' list as JSON and 'size_ratio' tuple as "max,min".
    Skips any None entries.
    """
    fieldnames = [
        'job_index',
        'mode',
        'time_window',
        'old_commit',
        'current_commit',
        'coverage',
        'groups',
        'total_constraints',
        'evenness',
        'size_ratio',
        'time_elapsed_seconds',
        'commit_count',
        'config_change_pct',
        'per_config_pct'
    ]

    # Guard: ensure directory exists, even if path is just a filename
    out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Writing {len(results)} result rows to '{csv_path}'")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # type: ignore
        writer.writeheader()

        for result in results:
            if not result:
                continue
            row = result.copy()
            row['groups'] = json.dumps(row['groups'])
            if row['size_ratio'][0] is not None:
                row['size_ratio'] = f"{row['size_ratio'][0]},{row['size_ratio'][1]}"
            else:
                row['size_ratio'] = ''
            row['config_change_pct'] = row.get('config_change_pct')
            row['per_config_pct'] = json.dumps(row.get('per_config_pct', {}))
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
        '--mode', choices=['original', 'krepairDC', 'defconfig'], default='original',
        help='Repair mode: original for krepair, krepairDC for krepairDC, defconfig for non-repaired defconfig .config coverage'
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
    commits = load_commits(commit_list_file)
    commits = commits[:num_kernels]
    copied_kernels = copy_kernel_multiple_times(kernels_src, tmp_dir, commits)

    # Build a list of jobs: (kernel_dir, time_window, job_index, mode, ...) for each kernel
    jobs = [
        (kernel_dir, time_window, commit_idx, mode,
         commit_list_file, old_commit_list_file)
        for commit_idx, kernel_dir in copied_kernels
    ]

    print(f"[INFO] Scheduling {len(jobs)} jobs in mode='{mode}'")
    for commit_idx, kernel_dir in copied_kernels:
        print(f"[INFO]  • Job {commit_idx}: kernel dir = {kernel_dir}")

    # Process the jobs in parallel
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(process_kernel, job) for job in jobs]
    results = []
    for fut in tqdm(as_completed(futures),
                    total=len(futures),
                    desc=f"[{mode}] jobs",
                    unit="job"):
        results.append(fut.result())

    results.sort(key=lambda r: r['job_index'])  # just in case

    # Write the results to CSV
    write_results_to_csv(results, output_csv)

    # Output the result summary
    success_count = sum(bool(r) for r in results if r is not None)
    print(f"Done ({args.time_window}, {mode}). {success_count} runs succeeded.")

if __name__ == '__main__':
    main()