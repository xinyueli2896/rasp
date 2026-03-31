
from __future__ import annotations

import os
import sys
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_seed(seed: int, base_args: list[str], ckpt_dir: str, ckpt_stem: str) -> float:
    seed_ckpt = os.path.join(ckpt_dir, f'{ckpt_stem}_seed{seed}')
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'train_yinyang.py'),
        '--seed',      str(seed),
        '--ckpt_name', seed_ckpt.replace(ckpt_dir + os.sep, ''),
        '--ckpt_dir',  ckpt_dir,
    ] + base_args

    print(f'\n{"="*60}')
    print(f'Seed {seed}  →  checkpoint: {seed_ckpt}.pt')
    print(f'{"="*60}')

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'[seed {seed}] Training failed with exit code {result.returncode}')
        return float('nan')

    # Parse best test accuracy from the last "Best test accuracy" line
    # Re-run with captured output just to grab the number
    result2 = subprocess.run(cmd + ['--epochs', '0'], capture_output=True, text=True)
    # Fallback: read it from the checkpoint name log above; we'll parse stdout instead
    # Actually just run normally and grep — re-run is wasteful, so parse from file.
    # Since subprocess already printed live, we need another approach: save acc to file.
    acc_file = seed_ckpt + '_acc.txt'
    if os.path.exists(acc_file):
        with open(acc_file) as f:
            return float(f.read().strip())
    return float('nan')


def run_seed_capturing(seed: int, base_args: list[str], ckpt_dir: str, ckpt_stem: str) -> float:
    seed_name = f'{ckpt_stem}_seed{seed}'
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'train_yinyang.py'),
        '--seed',      str(seed),
        '--ckpt_name', seed_name,
        '--ckpt_dir',  ckpt_dir,
    ] + base_args

    print(f'\n{"="*60}')
    print(f'Seed {seed}  →  {ckpt_dir}/{seed_name}.pt')
    print(f'{"="*60}', flush=True)

    best_acc = float('nan')
    import subprocess, re
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
        m = re.search(r'Best test accuracy on unseen starters:\s*([0-9.]+)', line)
        if m:
            best_acc = float(m.group(1))
    proc.wait()
    return best_acc


def multirun(args):
    seeds    = list(range(args.n_seeds))
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Forward all extra args to train_yinyang.py (everything except multirun-specific ones)
    base_args = args.train_args  # passed via -- separator

    results = []
    for seed in seeds:
        acc = run_seed_capturing(seed, base_args, ckpt_dir, args.ckpt_stem)
        results.append(acc)
        print(f'\n  [seed {seed}] best_test_acc = {acc:.4f}')

    valid = [r for r in results if not np.isnan(r)]
    print(f'\n{"="*60}')
    print(f'Multi-run summary  ({len(valid)}/{args.n_seeds} seeds succeeded)')
    print(f'{"="*60}')
    for seed, acc in zip(seeds, results):
        flag = '(FAILED)' if np.isnan(acc) else ''
        print(f'  seed {seed}: {acc:.4f}  {flag}')
    if valid:
        mean = np.mean(valid)
        std  = np.std(valid)
        print(f'\n  mean ± std : {mean:.4f} ± {std:.4f}')
        print(f'  min / max  : {min(valid):.4f} / {max(valid):.4f}')
    print(f'{"="*60}')


def get_args():
    p = argparse.ArgumentParser(
        description='Run train_yinyang.py over multiple seeds and report mean ± std',
        epilog='Pass train_yinyang.py args after --: e.g. %(prog)s --n_seeds 5 -- --n_skip 1 --epochs 200',
    )
    p.add_argument('--n_seeds',   type=int, default=5,
                   help='Number of seeds to run (seeds 0, 1, ..., n_seeds-1)')
    p.add_argument('--ckpt_dir',  type=str, default='checkpoints')
    p.add_argument('--ckpt_stem', type=str, default='yinyang_multirun',
                   help='Checkpoint name stem; per-seed files saved as <stem>_seed<N>.pt')
    p.add_argument('train_args',  nargs=argparse.REMAINDER,
                   help='Arguments forwarded to train_yinyang.py (after --)')
    args = p.parse_args()
    # Strip leading '--' separator if present
    if args.train_args and args.train_args[0] == '--':
        args.train_args = args.train_args[1:]
    return args


if __name__ == '__main__':
    multirun(get_args())
