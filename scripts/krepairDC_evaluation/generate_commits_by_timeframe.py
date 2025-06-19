#!/usr/bin/env python3
"""
Script to read a list of commit hashes and, for each one, find the ancestor commit a specified timeframe behind.
Writes the resulting commits into an output text file, one per line.

Usage:
  python generate_commits_by_timeframe.py \
      --input commits.txt \
      --timeframe 12h|72h|7d \
      --r /path/to/repo \
      [--output behind_commits.txt]
"""
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

def get_commit_at_time(repo_path: Path, base_commit: str, rel_time: timedelta) -> str:
    """
    Find ancestor commit that is at least rel_time older, following mainline only.
    """
    # Get the date/time of the base commit
    date_str = subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', base_commit],
        cwd=repo_path, text=True
    ).strip()
    base_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    target_date = base_date - rel_time

    # Walk MAINLINE ancestors only with --first-parent
    log = subprocess.check_output(
        ['git', 'log', '--first-parent', '--pretty=%H %cI', base_commit],
        cwd=repo_path, text=True
    )

    chosen = None
    for line in log.splitlines():
        sha, commit_date_str = line.split(' ', 1)
        commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
        if commit_date <= target_date:
            chosen = sha
            break

    return chosen or base_commit


def get_commit_info(repo_path: Path, commit: str) -> Tuple[str, str]:
    """
    Get commit date and first line of commit message.
    """
    info = subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI %s', commit],
        cwd=repo_path, text=True
    ).strip()
    date_str, subject = info.split(' ', 1)
    return date_str, subject


def count_mainline_commits(repo_path: Path, start: str, end: str) -> int:
    """
    Count commits between start and end following mainline only.
    """
    count = subprocess.check_output(
        ['git', 'rev-list', '--count', '--first-parent', f'{start}..{end}'],
        cwd=repo_path, text=True
    ).strip()
    return int(count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a list of ancestor commits a given timeframe behind each listed commit."
    )
    parser.add_argument('-r', '--repo', type=Path, default=Path('.'),
                        help='Path to the Git repository (default: current directory)')
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Text file with one commit hash per line')
    parser.add_argument('-t', '--timeframe', choices=['12h', '72h', '7d'], required=True,
                        help='Timeframe behind each commit to find (12h, 72h, or 7d)')
    parser.add_argument('-o', '--output', type=Path, default=Path('behind_commits.txt'),
                        help='Output file to write the resulting commits')
    args = parser.parse_args()

    tf_map = {'12h': timedelta(hours=12),
              '72h': timedelta(hours=72),
              '7d':  timedelta(days=7)}
    rel_time = tf_map[args.timeframe]

    repo = args.repo.resolve()
    if not repo.is_dir():
        parser.error(f"Path {repo} is not a valid directory")
    if not (repo / '.git').exists():
        parser.error(f"Directory {repo} is not a Git repository")
    if not args.input.exists():
        parser.error(f"Input file {args.input} does not exist")

    commits = [l.strip() for l in args.input.read_text().splitlines() if l.strip()]
    if not commits:
        parser.error(f"No commits found in {args.input}")

    results = []
    print(f"\n{'='*80}")
    print(f"Finding commits {args.timeframe} behind (following mainline only)")
    print(f"{'='*80}\n")

    for sha in commits:
        try:
            behind = get_commit_at_time(repo, sha, rel_time)
            results.append(behind)

            # Get info for both commits
            end_date, end_subject = get_commit_info(repo, sha)
            start_date, start_subject = get_commit_info(repo, behind)

            # Count mainline commits between them
            if behind != sha:
                mainline_count = count_mainline_commits(repo, behind, sha)
            else:
                mainline_count = 0

            # Pretty print the results
            print(f"Original: {sha[:12]} ({end_date})")
            print(f"          {end_subject[:70]}")
            print(f"↓")
            print(f"Behind:   {behind[:12]} ({start_date})")
            print(f"          {start_subject[:70]}")
            print(f"")
            print(f"Mainline commits between: {mainline_count}")
            print(f"{'-'*80}\n")

        except subprocess.CalledProcessError as e:
            print(f"Error processing {sha}: {e}")
            results.append('')

    args.output.write_text('\n'.join(results) + '\n')
    print(f"Wrote {len(results)} commits to {args.output}")

if __name__ == '__main__':
    main()