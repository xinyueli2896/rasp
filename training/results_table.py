#!/usr/bin/env python3
"""Assemble the results table from the saved checkpoints.

    python -m training.results_table
    python -m training.results_table --latex

Reads checkpoints/*.pt rather than scrollback. Every training script stores
its split and its per-cell RFA in the checkpoint, so the table cannot drift
from the runs that produced it, and a checkpoint trained under a different
split is refused rather than silently averaged in.

    mean +- std over seeds, matching report.tex's "+- = std over 5 seeds".

With one seed the std column is blank rather than 0.000, because a single
run has no spread and printing zero would imply otherwise.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.integer_dataset import validate_split  # noqa: E402
from training.pretrain_integer import CELLS      # noqa: E402

# (glob, tag, description, always_show)
# always_show=False rows appear only when they actually match something, so
# the table is not padded with rows that will never exist.
ROWS = [
    ('ar_integer_seed*.pt',               'Pre',  'AR alone, trained on pool A',   True),
    ('integer_finetune_*seed*.pt',        'FT',   'AR continued on B, no adapter', True),
    ('integer_adapter-frozen_*seed*.pt',  'YY',   'adapter, AR frozen',            True),
    ('integer_adapter-lora*_*seed*.pt',   'YY+L', 'adapter + LoRA on the AR',      True),
    ('integer_adapter-trainar_*seed*.pt', 'YY+T', 'adapter + AR fully trainable',  True),
    # Checkpoints written before the variant suffix existed, when every
    # adapter run was `integer_adapter_<slice>_seed<N>.pt` regardless of
    # whether the AR was frozen. Listed separately rather than folded into
    # YY because which variant produced them is no longer recoverable from
    # the filename. The globs do not overlap -- this one needs an UNDERSCORE
    # after "adapter" where the others have a hyphen.
    ('integer_adapter_*seed*.pt',         'YY?',  'adapter, variant unknown '
                                                  '(pre-rename checkpoint)',       False),
]


def collect(pattern, ckpt_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(ckpt_dir, pattern))):
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if 'rfa' not in blob:
            continue
        runs.append((path, blob))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', type=str, default='checkpoints')
    p.add_argument('--latex', action='store_true')
    a = p.parse_args()

    ref = validate_split()
    print(f'split   A={ref["pretrain"]}  B={ref["adapter"]}')
    for k in CELLS:
        print(f'  {k:<14} {ref["cells"][k]}')
    print()

    hdr = f'{"":<5}{"model":<40}' + ''.join(f'{k:>18}' for k in CELLS) + f'{"n":>4}'
    print(hdr); print('-' * len(hdr))

    table = {}
    for pattern, tag, desc, always in ROWS:
        runs = collect(pattern, a.ckpt_dir)
        if not runs:
            if always:
                print(f'{tag:<5}{desc:<40}' + f'{"(no checkpoints)":>18}')
            continue

        stale = [os.path.basename(p) for p, b in runs
                 if b.get('split', {}).get('cells') != ref['cells']]
        if stale:
            print(f'{tag:<5}{desc:<40}  REFUSED: trained under a different '
                  f'split: {stale}')
            print(f'       re-run them, or the row averages incomparable '
                  f'numbers')
            continue

        vals = {k: np.array([b['rfa'][k] for _, b in runs]) for k in CELLS}
        table[tag] = vals
        cells = ''
        for k in CELLS:
            v = vals[k]
            cells += (f'{v.mean():>11.3f}       ' if len(v) == 1
                      else f'{v.mean():>11.3f}+-{v.std(ddof=1):<5.3f}')
        print(f'{tag:<5}{desc:<40}{cells}{len(runs):>4}')

    print(f'\n  RFA: generate 31 tokens from the one-token prompt [x], greedy. '
          f'chance = {1/12:.3f}')
    print('  pretrain_only is the forgetting cell -- FT is expected to collapse')
    print('  there, the adapter cannot (the AR weights never move).')
    print('  unseen is the headline; neither pool contains {6, 8}.')
    if any(len(v[CELLS[0]]) == 1 for v in table.values()):
        print('\n  std blank = a single seed. Run at least 5 before quoting '
              'these; the unseen')
        print('  cell is 62 predictions and seeds are the only source of '
              'spread.')

    if a.latex and table:
        print('\n% ---- LaTeX ----')
        print('\\begin{tabular}{ll' + 'c' * len(CELLS) + '}')
        print('\\toprule')
        print(' & '.join(['', 'Model'] + [c.replace('_', '-') for c in CELLS])
              + ' \\\\')
        print('\\midrule')
        for pattern, tag, desc, _ in ROWS:
            if tag not in table:
                continue
            cs = []
            for k in CELLS:
                v = table[tag][k]
                cs.append(f'{v.mean():.3f}' if len(v) == 1
                          else f'{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}')
            print(' & '.join([tag, desc] + cs) + ' \\\\')
        print('\\bottomrule\n\\end{tabular}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
