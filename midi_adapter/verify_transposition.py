"""
Verify that transposed windows in a .pt training dataset are self-consistent:
for each window, the ACTUAL chord progression detected in its CP tensor should
match the I-IV-V-I sequence in the key that was PAIRED with it.

For window i:
  paired_key   = keys.pt[i]                                (int in 0..11)
  expected     = [(paired_key + OFFSETS[c%4]) % 12 for c in range(N_chords)]
  detected     = chromagram-argmax-major-triad on cp[i]     per half-bar
  agreement    = fraction of chord slots where expected == detected

If transposition applied correctly and the paired key label is right, the
per-window agreement should be very high (1.0 for windows that were 0-wrong
before repair; ≥ 0.75 for --allow_wrong 2 windows because the wrong slots
were repaired to match).

Usage
-----
    python -m midi_adapter.verify_transposition \\
        --pt        /l/users/xinyue.li/data/pop909_ivvi_w1_seen.pt \\
        --window_len 64 \\
        --chords_per_bar 2

Prints per-key mean agreement and a histogram of per-window agreement,
plus a couple of failing examples for manual inspection.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.filter_nottingham import (
    _extract_chord_roots_from_cp,
    ROOT_NAMES,
)
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS


def _agreement(cp_window: np.ndarray, key: int, chords_per_bar: int) -> tuple[float, list[int], list[int]]:
    detected = _extract_chord_roots_from_cp(cp_window, chords_per_bar=chords_per_bar)
    n_chords = len(detected)
    expected = [(key + OFFSETS[c % 4]) % 12 for c in range(n_chords)]
    valid = [(d, e) for d, e in zip(detected, expected) if d >= 0]
    if not valid:
        return 0.0, detected, expected
    n_match = sum(1 for d, e in valid if d == e)
    return n_match / len(valid), detected, expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pt',         required=True,
                   help='Path to a training dataset .pt (e.g. pop909_ivvi_w1_seen.pt).')
    p.add_argument('--window_len',    type=int, default=64)
    p.add_argument('--chords_per_bar', type=int, default=2, choices=[1, 2, 4])
    p.add_argument('--sample',        type=int, default=0,
                   help='Cap windows to check (0 = all).')
    p.add_argument('--n_fail_examples', type=int, default=4,
                   help='Print this many worst-agreement windows for inspection.')
    args = p.parse_args()

    data = torch.load(args.pt, weights_only=True).numpy()
    if data.ndim != 2:
        raise SystemExit(f'expected (N, subseq) tensor, got {data.shape}')
    n_rows, subseq = data.shape
    if n_rows % args.window_len != 0:
        raise SystemExit(f'{n_rows} rows not divisible by window_len={args.window_len}')
    n_windows = n_rows // args.window_len
    windows = data.reshape(n_windows, args.window_len, subseq)

    keys_path = args.pt[:-3] + '.keys.pt'
    if not os.path.exists(keys_path):
        raise SystemExit(f'no {keys_path} — verification needs the paired key labels')
    keys = torch.load(keys_path, weights_only=True).long().tolist()
    assert len(keys) == n_windows, f'{len(keys)} keys vs {n_windows} windows'

    if args.sample and args.sample < n_windows:
        idx = np.random.default_rng(0).choice(n_windows, size=args.sample, replace=False)
    else:
        idx = np.arange(n_windows)

    per_key: dict[int, list[float]] = {}
    all_scores: list[tuple[int, float, int]] = []   # (i, agreement, key)
    for i in idx:
        i = int(i)
        agr, det, exp = _agreement(windows[i], keys[i], args.chords_per_bar)
        per_key.setdefault(keys[i], []).append(agr)
        all_scores.append((i, agr, keys[i]))

    print(f'Verifying transposition consistency on {len(idx)} windows from {args.pt}\n')
    print(f'  {"key":<4}  {"n":>5}  {"mean_agree":>10}  {"< 0.5 frac":>10}')
    total_scores: list[float] = []
    for k in sorted(per_key):
        scores = np.array(per_key[k])
        total_scores.extend(scores.tolist())
        print(f'  {ROOT_NAMES[k]:<4}  {len(scores):>5}  {scores.mean():>10.3f}  '
              f'{(scores < 0.5).mean():>10.3f}')
    total = np.array(total_scores)
    print(f'  {"MEAN":<4}  {"":>5}  {total.mean():>10.3f}  '
          f'{(total < 0.5).mean():>10.3f}')

    # Histogram of per-window agreement.
    bins = np.linspace(0.0, 1.0, 6)   # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, edges = np.histogram(total, bins=bins)
    print('\n  Per-window agreement histogram:')
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = '█' * (h * 50 // max(hist.max(), 1))
        print(f'    [{lo:.1f}, {hi:.1f})  {h:>6}  {bar}')

    # Worst offenders — show what went wrong.
    if args.n_fail_examples:
        worst = sorted(all_scores, key=lambda x: x[1])[:args.n_fail_examples]
        print(f'\n  Worst {args.n_fail_examples} windows:')
        for i, agr, k in worst:
            _, det, exp = _agreement(windows[i], k, args.chords_per_bar)
            det_names = [ROOT_NAMES[d] if d >= 0 else '-' for d in det]
            exp_names = [ROOT_NAMES[e] for e in exp]
            print(f'\n    window {i}  key={ROOT_NAMES[k]}  agreement={agr:.3f}')
            print(f'      expected : ' + '  '.join(f'{e:>3}' for e in exp_names))
            print(f'      detected : ' + '  '.join(f'{d:>3}' for d in det_names))


if __name__ == '__main__':
    main()
