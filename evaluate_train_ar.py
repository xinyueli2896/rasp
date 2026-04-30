#!/usr/bin/env python3
"""
Evaluate all train_ar experiment checkpoints.

For each dataset config (2to16 / 0to16 / 0to16_plus), compares:
  Pretrain   : ar_pretrain_0to6  (AR only, no adapter)
  Finetune   : ar_finetuned_<config>  (finetuned AR, no adapter)
  FrozenAR   : adapter_<config>_seed*  (frozen AR + adapter)
  TrainAR    : adapter_train_ar_<config>_seed*  (trainable AR init from pretrain + adapter)
  ScratchAR  : adapter_train_ar_scratch_<config>_seed*  (AR from scratch + adapter)

Usage:
  python evaluate_train_ar.py
  python evaluate_train_ar.py --configs 2to16 0to16
  python evaluate_train_ar.py --verbose
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (
    load_pretrain,
    load_finetune,
    _load_seed_models,
    _eval_models,
    rule_following_acc,
    _fmt,
)
from data.dataset import (
    EVAL_PRETRAIN_ONLY,
    EVAL_FINETUNE_ONLY,
    EVAL_BOTH,
    EVAL_NEITHER,
)


ALL_CONFIGS = [
    ('2to16',      'ar_finetuned_2to16'),
    ('0to16',      'ar_finetuned_0to16'),
    ('0to16_plus', 'ar_finetuned_0to16_plus'),
]

CATEGORIES = [
    ('Pretrain-only', EVAL_PRETRAIN_ONLY, 'A\\B={0,1}'),
    ('Finetune-only', EVAL_FINETUNE_ONLY, 'B\\A={6..15}'),
    ('Both          ', EVAL_BOTH,         'A∩B={2..5}'),
    ('Neither       ', EVAL_NEITHER,      '{17,19,20,21}'),
]

ALL_STARTERS = EVAL_PRETRAIN_ONLY + EVAL_FINETUNE_ONLY + EVAL_BOTH + EVAL_NEITHER


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def eval_config(config: str, ft_stem: str, pretrain_model, args, device):
    print(f'\n{"="*80}')
    print(f'CONFIG: {config}')
    print(f'{"="*80}')

    args.ft_ckpt = os.path.join(args.ckpt_dir, f'{ft_stem}.pt')
    ft_model = load_finetune(args, device)

    frozen_models   = _load_seed_models(f'adapter_{config}',                  args, device)
    train_ar_models = _load_seed_models(f'adapter_train_ar_{config}',         args, device)
    scratch_models  = _load_seed_models(f'adapter_train_ar_scratch_{config}', args, device)

    model_groups = [
        ('Pretrain',   [pretrain_model], False),
        ('Finetune',   [ft_model],       False),
        ('FrozenAR',   frozen_models,    len(frozen_models) > 1),
        ('TrainAR',    train_ar_models,  len(train_ar_models) > 1),
        ('ScratchAR',  scratch_models,   len(scratch_models) > 1),
    ]

    # Evaluate all groups across all categories
    results = {}
    for lbl, mdls, _ in model_groups:
        results[lbl] = {}
        for cat_label, starters, _ in CATEGORIES:
            per, mean, std = _eval_models(mdls, starters, args.n_cycles, args.prompt_len, device)
            results[lbl][cat_label] = (per, mean, std)
        per, mean, std = _eval_models(mdls, ALL_STARTERS, args.n_cycles, args.prompt_len, device)
        results[lbl]['Overall'] = (per, mean, std)

    # Print summary table
    col_w    = 14
    col_w_ms = 17   # wider to fit mean±std

    all_labels = [lbl for lbl, _, _ in model_groups]
    is_multi   = {lbl: multi for lbl, _, multi in model_groups}

    header = f"{'Data Split':<20} {'Starters':<18}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        header += f" {lbl:>{w}}"
    sep = '-' * len(header)

    print(f'\n  Seed counts:  FrozenAR={len(frozen_models)}  '
          f'TrainAR={len(train_ar_models)}  ScratchAR={len(scratch_models)}')
    print()
    print(header)
    print(sep)
    for cat_label, _, starters_str in CATEGORIES:
        row = f"{cat_label:<20} {starters_str:<18}"
        for lbl in all_labels:
            w = col_w_ms if is_multi[lbl] else col_w
            _, mean, std = results[lbl][cat_label]
            row += ' ' + _fmt(mean, std, w, is_multi[lbl])
        print(row)
    print(sep)
    row = f"{'Overall':<20} {'all 16 starters':<18}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        _, mean, std = results[lbl]['Overall']
        row += ' ' + _fmt(mean, std, w, is_multi[lbl])
    print(row)
    print('=' * len(header))

    # Per-seed detail
    if args.verbose:
        cat_labels_all = [c for c, _, _ in CATEGORIES] + ['Overall']
        starters_all   = [s for _, s, _ in CATEGORIES] + [ALL_STARTERS]
        for lbl, mdls, multi in model_groups:
            if not multi:
                continue
            print(f'\n  Per-seed breakdown: {lbl}')
            col_names = '  '.join(f'{c[:14]:>14}' for c in cat_labels_all)
            print(f"    {'seed':>6}  {col_names}")
            print(f"    {'-'*6}  {'-'*len(col_names)}")
            for i, mdl in enumerate(mdls):
                accs = []
                for starters in starters_all:
                    _, mean = rule_following_acc(
                        mdl, starters, args.n_cycles, args.prompt_len, device)
                    accs.append(mean)
                row = f"    {i:>6}  " + '  '.join(f'{a:>14.4f}' for a in accs)
                print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading pretrain model ...')
    pretrain_model = load_pretrain(args, device)

    selected = [(c, ft) for c, ft in ALL_CONFIGS if c in args.configs]
    if not selected:
        print(f'No matching configs found. Available: {[c for c, _ in ALL_CONFIGS]}')
        return

    for config, ft_stem in selected:
        eval_config(config, ft_stem, pretrain_model, args, device)

    print('\n\n=== EVALUATION COMPLETE ===')


def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate train_ar experiment checkpoints',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--ar_ckpt',    type=str, default='checkpoints/ar_pretrain_0to6.pt',
                   help='Pretrain AR checkpoint')
    p.add_argument('--ft_ckpt',    type=str, default='checkpoints/ar_finetuned_2to16.pt',
                   help='Finetune checkpoint (overridden per config; kept for API compat)')
    p.add_argument('--ckpt_dir',   type=str, default='checkpoints')
    p.add_argument('--d_model',    type=int, default=128)
    p.add_argument('--n_layers',   type=int, default=4)
    p.add_argument('--n_heads',    type=int, default=4)
    p.add_argument('--n_cycles',   type=int, default=8)
    p.add_argument('--prompt_len', type=int, default=1)
    p.add_argument('--verbose',    action='store_true',
                   help='Print per-seed breakdown for each multi-seed variant')
    p.add_argument('--force_fallback', action='store_true', default=True)
    p.add_argument('--configs',    type=str, nargs='*',
                   default=['2to16', '0to16', '0to16_plus'],
                   help='Which dataset configs to evaluate (default: all 3)\n'
                        'Choices: 2to16  0to16  0to16_plus')
    return p.parse_args()


if __name__ == '__main__':
    run(get_args())
