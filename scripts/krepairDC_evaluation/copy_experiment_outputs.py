#!/usr/bin/env python3
"""
copy_experiment_outputs.py

Copy experiment log and result files out of a large kernel-clone tree so
you can delete the kernels themselves afterward.

Directory structure assumed:
    {base_dir}/{mode}/{idx}_{commit}/...
where mode is one of: defconfig, krepair, krepairDC.

Usage:
    python3 copy_experiment_outputs.py \
        --mode krepairDC \
        --output-dir /path/to/output \
        [--base-dir /tmp/testing_kernels]
"""

import argparse
import shutil
from pathlib import Path
import sys
from datetime import datetime

def gather_patterns(mode):
    if mode == 'defconfig':
        return [
            'defconfig_koverage.log',
            'defconfig_make.log',
            'patchset_*.diff',
            'total_coverage.log',
            'defconfig_coverage_results.json',
            'patch_coverage.log',
        ]
    else:  # krepair or krepairDC
        summary = 'krepairDC_summary.csv' if mode == 'krepairDC' else 'krepair_summary.csv'
        return [
            '*-x86_64.config',
            '*_koverage.log',
            '*_coverage_results.json',
            'total_coverage_results.json',
            'defconfig_make.log',
            'patchset_*.diff',
            'total_coverage.log',
            'patch_coverage.log',
            'config_change_percentage.txt',
            summary,
        ]

def copy_for_mode(base_dir: Path, mode: str, output_dir: Path):
    src_root = base_dir / mode
    if not src_root.is_dir():
        print(f"[ERROR] mode directory not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    # create a timestamped output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_output_dir = output_dir / f"{mode}_{timestamp}"

    patterns = gather_patterns(mode)

    for sub in sorted(src_root.iterdir()):
        if not sub.is_dir():
            continue

        rel_name = sub.name  # e.g. "0_ab12cd3"
        dest_dir = mode_output_dir / rel_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for pat in patterns:
            matches = list(sub.glob(pat))
            if not matches:
                print(f"[WARN] no files matching '{pat}' in {sub}")
            for src in matches:
                try:
                    shutil.copy2(src, dest_dir)
                    print(f"Copied {src.name} → {dest_dir}/")
                except Exception as e:
                    print(f"[ERROR] failed to copy {src}: {e}", file=sys.stderr)

def main():
    p = argparse.ArgumentParser(
        description="Copy experiment result files out of kernel clones."
    )
    p.add_argument(
        '--base-dir', '-b',
        type=Path,
        default=Path('/tmp/testing_kernels'),
        help="Root of your testing_kernels tree (default: /tmp/testing_kernels)"
    )
    p.add_argument(
        '--mode', '-m',
        choices=['defconfig', 'krepair', 'krepairDC'],
        required=True,
        help="Which mode subdirectory to process"
    )
    p.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help="Where to mirror all the logs/results"
    )
    args = p.parse_args()

    copy_for_mode(args.base_dir, args.mode, args.output_dir)

if __name__ == '__main__':
    main()