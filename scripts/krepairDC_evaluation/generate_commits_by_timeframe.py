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


# Helper: robust ISO-8601 → datetime parser
# Git’s “%cI” ends with either “Z” or an explicit ±HH:MM offset.  We normalise
# the Z-suffix so strptime can consume it.
def _parse_iso8601(ts: str) -> datetime:
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    return datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S%z')


def get_commit_at_time(repo_path: Path, base_commit: str, rel_time: timedelta) -> str:
    """
    Given a repo and a commit hash, find the most recent ancestor commit
    that is at least `rel_time` older than the base commit’s author date.
    If none qualify, return the original commit hash.
    """
    # Date of the base commit
    date_str = subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', base_commit],
        cwd=repo_path, text=True
    ).strip()
    base_date = _parse_iso8601(date_str)
    target_date = base_date - rel_time

    # Let Git stop at the first ancestor that satisfies the cutoff
    cutoff = target_date.strftime('%Y-%m-%dT%H:%M:%S%z')
    chosen = subprocess.check_output(
        ['git', 'rev-list', '-n1', '--before', cutoff, base_commit],
        cwd=repo_path, text=True
    ).strip()

    return chosen or base_commit   # rev-list prints nothing if none qualify


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a list of ancestor commits a given timeframe behind each listed commit."
    )
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

    if not (Path('.') / '.git').exists():
        parser.error("Current working directory is not a Git repository")
    if not args.input.exists():
        parser.error(f"Input file {args.input} does not exist")

    commits = [l.strip() for l in args.input.read_text().splitlines() if l.strip()]
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

    args.output.write_text('\n'.join(results) + '\n')
    print(f"Wrote {len(results)} commits to {args.output}")


if __name__ == '__main__':
    main()