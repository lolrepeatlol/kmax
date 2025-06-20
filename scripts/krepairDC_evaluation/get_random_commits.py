#!/usr/bin/env python3
"""
get_random_commits.py  –  Sample N random mainline Linux commits.

Usage:
    python get_random_commits.py [options]
Options:
    --repo <path>      Path to the git repository (default: current directory)
    --count <N>        Number of commits to sample (default: 385)
    --since <time>     Only consider commits since this time (default: '3 years ago')
    --output <file>    Output file for sampled commit SHAs (default: 'random_commits.txt')
    --seed <N>         RNG seed for reproducibility (optional)
"""

import argparse, random, subprocess, sys, time, os, tempfile, shutil, pathlib

MAX_FETCHES   = 5        # fetch up to this many times if history is too shallow
MAX_RETRIES   = 3        # repeat the whole git-log step on transient failure
FETCH_DEEPEN  = 100000   # lines of history to add on each fetch – large is fastest

def run(cmd, **kw):
    """Thin wrapper: return stdout, raise on non-zero."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}: {r.stderr.strip()}")
    return r.stdout

def ensure_history(repo, need):
    """Make sure <repo> has at least <need> commits reachable from origin/master."""
    for attempt in range(MAX_FETCHES + 1):
        got = int(run(['git', '-C', repo, 'rev-list', '--count', '--first-parent', 'origin/master']))
        if got >= need:
            return
        print(f"[INFO] Only {got} commits; fetching to deepen history…")
        run(['git', '-C', repo, 'fetch', '--quiet', '--deepen', str(FETCH_DEEPEN)])
    raise RuntimeError(f"After {MAX_FETCHES} fetches there are still < {need} commits.")

def sample_commits(repo, since, count):
    """Return <count> unique SHAs since <since>."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = run(['git', '-C', repo, 'log', '--first-parent', '--pretty=%H', '--since', since, 'origin/master'])
            shas = out.strip().splitlines()
            if len(shas) < count:
                raise RuntimeError(f"Found {len(shas)} commits (< {count}); need more history.")
            return random.sample(shas, count)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"[WARN] git log failed (try {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.', help='Path to torvalds/linux clone')
    ap.add_argument('--count', type=int, default=385, help='Commits to sample')
    ap.add_argument('--since', default='3 years ago', help="git --since syntax")
    ap.add_argument('--output', default='random_commits.txt', help='Destination file')
    ap.add_argument('--seed', type=int, help='RNG seed for reproducibility')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 1. Confirm repo exists & has enough reachable history
    if not pathlib.Path(args.repo, '.git').is_dir():
        sys.exit(f"[ERR] No git repo at {args.repo}")
    ensure_history(args.repo, args.count)

    # 2. Grab commits
    commits = sample_commits(args.repo, args.since, args.count)
    random.shuffle(commits)

    # 3. Write to temp-file, verify, then atomically move into place
    tmp_fd, tmp_name = tempfile.mkstemp(prefix='commits_', text=True)
    with os.fdopen(tmp_fd, 'w') as f:
        f.write('\n'.join(commits) + '\n')

    # Verify
    written = sum(1 for _ in open(tmp_name, 'r'))
    if written != args.count:
        os.remove(tmp_name)
        sys.exit(f"[ERR] Sanity check failed: wrote {written} lines, expected {args.count}")

    shutil.move(tmp_name, args.output)
    print(f"✓ Wrote {args.count} commit SHAs to {args.output}")

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.exit(f"[FATAL] {exc}")