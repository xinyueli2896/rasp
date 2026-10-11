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
    ('ar_integer_seed*.pt',               'Pre',  'AR alone, trained on pool A',   True, None),
    ('integer_finetune_*seed*.pt',        'FT',   'AR continued on B, no adapter', True, None),
    ('integer_adapter-frozen_*seed*.pt',  'YY',   'adapter, AR frozen',            True, None),
    ('integer_adapter-lora*_*seed*.pt',   'YY+L', 'adapter + LoRA on the AR',      True, None),
    ('integer_adapter-trainar_*seed*.pt', 'YY+T', 'adapter + AR fully trainable',  True, None),
    # Single-stage runs on A u B. Only `unseen` is held out for these, so the
    # other three cells are in-distribution and NOT comparable with the rows
    # above, where they are held out by construction.
    ('integer_adapter-frozen-enc*_*seed*.pt', 'YY-E',
     'adapter, NO rule input (encoder)',                           True, '-sup'),
    ('integer_adapter-frozen-enc*-sup*_*seed*.pt', 'YY-ES',
     'adapter, NO rule input + CE supervision',                     True, None),
    ('integer_scratch_*seed*.pt',         'S',    'AR from scratch on AuB, no rule', True, None),
    ('integer_joint_scratch_*seed*.pt',   'S+YY', 'AR from scratch + adapter, joint', True, None),
    # Checkpoints written before the variant suffix existed, when every
    # adapter run was `integer_adapter_<slice>_seed<N>.pt` regardless of
    # whether the AR was frozen. Listed separately rather than folded into
    # YY because which variant produced them is no longer recoverable from
    # the filename. The globs do not overlap -- this one needs an UNDERSCORE
    # after "adapter" where the others have a hyphen.
    ('integer_adapter_*seed*.pt',         'YY?',  'adapter, variant unknown '
                                                  '(pre-rename checkpoint)',  False, None),
]


def collect(pattern, ckpt_dir, exclude=None, ref_cells=None):
    """-> (runs, n_other_split).

    `exclude` drops paths containing that substring, because fnmatch has no
    negation and the supervised encoder name is a superstring of the
    unsupervised one.

    Checkpoints whose stored split differs from the reference are COUNTED
    and dropped rather than refusing the whole row. The split sweep writes
    tagged files (integer_adapter-frozen_a3o2_...) that match the plain
    pattern, and before this one sweep made the default YY row unreadable.
    Those belong to `python -m training.split_sweep`, not here.
    """
    runs, other = [], 0
    for path in sorted(glob.glob(os.path.join(ckpt_dir, pattern))):
        if exclude and exclude in os.path.basename(path):
            continue
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if 'rfa' not in blob:
            continue
        if ref_cells is not None and blob.get('split', {}).get('cells') != ref_cells:
            other += 1
            continue
        runs.append((path, blob))
    return runs, other


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

    hdr = f'{"":<6}{"model":<40}' + ''.join(f'{k:>18}' for k in CELLS) + f'{"n":>4}'
    print(hdr); print('-' * len(hdr))

    table = {}
    n_other_total = 0
    for pattern, tag, desc, always, excl in ROWS:
        runs, n_other = collect(pattern, a.ckpt_dir, excl, ref['cells'])
        n_other_total += n_other
        if not runs:
            if always:
                extra = (f'  ({n_other} under other splits)' if n_other else '')
                print(f'{tag:<6}{desc:<40}' + f'{"(no checkpoints)":>18}{extra}')
            continue

        vals = {k: np.array([b['rfa'][k] for _, b in runs]) for k in CELLS}
        table[tag] = vals
        cells = ''
        for k in CELLS:
            v = vals[k]
            cells += (f'{v.mean():>11.3f}       ' if len(v) == 1
                      else f'{v.mean():>11.3f}+-{v.std(ddof=1):<5.3f}')
        note = f'  +{n_other} other splits' if n_other else ''
        print(f'{tag:<6}{desc:<40}{cells}{len(runs):>4}{note}')

    print(f'\n  RFA: generate 31 tokens from the one-token prompt [x], greedy. '
          f'chance = {1/12:.3f}')
    print('  pretrain_only is the forgetting cell -- FT is expected to collapse')
    print('  there, the adapter cannot (the AR weights never move).')
    print('  unseen is the headline; neither pool contains {6, 8}.')
    print('  S / S+YY train on A u B in ONE stage, so for those rows only')
    print('  `unseen` is held out -- their other three cells are training data')
    print('  and must not be read across against Pre/FT/YY.')
    if n_other_total:
        print(f'\n  {n_other_total} checkpoint(s) trained under a DIFFERENT '
              f'split were excluded from every row above.')
        print('  Those are the split sweep\'s; read them with '
              '`python -m training.split_sweep`.')
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
        for pattern, tag, desc, _, _x in ROWS:
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
