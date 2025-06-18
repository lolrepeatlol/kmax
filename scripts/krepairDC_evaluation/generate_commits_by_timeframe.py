#!/usr/bin/env python3
"""
Script to read a list of commit hashes and, for each one, find the ancestor commit a specified timeframe behind.
Writes the resulting commits into an output text file, one per line.

Usage:
  python generate_commits_by_timeframe.py \
      --input commits.txt \
      --timeframe 12h|72h|7d \
      [--output behind_commits.txt]
Assumes you are running inside a git repo.
"""
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def get_commit_at_time(repo_path: Path, base_commit: str, rel_time: timedelta) -> str:
    """
    Given a repo and a commit hash, find the most recent ancestor commit
    that is older than or equal to (commit_date - rel_time).
    Returns the found commit hash (or the oldest if none meet the cutoff).
    """
    # Get the date/time of the base commit
    date_str = subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', base_commit],
        cwd=repo_path,
        text=True
    ).strip()
    base_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    target_date = base_date - rel_time

    # Walk ancestors of base_commit (including itself)
    log = subprocess.check_output(
        ['git', 'log', '--pretty=%H %cI', base_commit],
        cwd=repo_path,
        text=True
    )

    chosen = None
    for line in log.splitlines():
        sha, commit_date_str = line.split()
        commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
        # first commit older than or equal to target
        if commit_date <= target_date:
            chosen = sha
            break
    # fallback: use the last (oldest) in the log
    if chosen is None:
        parts = log.splitlines()
        chosen = parts[-1].split()[0] if parts else base_commit
    return chosen

def main():
    parser = argparse.ArgumentParser(
        description="Generate a list of ancestor commits a given timeframe behind each listed commit."
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Text file with one commit hash per line'
    )
    parser.add_argument(
        '--timeframe', '-t',
        choices=['12h', '72h', '7d'],
        required=True,
        help='Timeframe behind each commit to find (12h, 72h, or 7d)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('behind_commits.txt'),
        help='Output file to write the resulting commits'
    )
    args = parser.parse_args()

    # Map timeframe strings to timedeltas
    tf_map = {
        '12h': timedelta(hours=12),
        '72h': timedelta(hours=72),
        '7d' : timedelta(days=7),
    }
    rel_time = tf_map[args.timeframe]

    if not (Path('.') / '.git').exists():
        parser.error("Current working directory is not a Git repository")
    if not args.input.exists():
        parser.error(f"Input file {args.input} does not exist")

    commits = [line.strip() for line in args.input.read_text().splitlines() if line.strip()]
    if not commits:
        parser.error(f"No commits found in {args.input}")

    results = []
    for sha in commits:
        try:
            behind = get_commit_at_time(Path('.'), sha, rel_time)
            results.append(behind)
            print(f"{sha} → {behind}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {sha}: {e}")
            results.append('')

    # Write results
    args.output.write_text("\n".join(results) + "\n")
    print(f"Wrote {len(results)} commits to {args.output}")

if __name__ == '__main__':
    main()