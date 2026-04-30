#!/usr/bin/env python3
"""
Evaluate overlap experiment checkpoints.

For each config, loads:
  {stem}_pretrain.pt      — pretrained AR (AutoregressiveTransformer)
  {stem}_finetune.pt      — finetuned AR
  {stem}_adapter_seed*.pt — adapter x5 seeds (frozen pretrained AR)

Eval categories per config:
  A\B         : starters seen only during pretrain     (varies per config)
  B\A         : starters seen only during finetune     (varies per config)
  A∩B         : starters seen in both                  (varies per config)
  Never-seen  : {17,19,20,21}                          (fixed, always same)
  Overall     : A∪B∪{17,19,20,21}

Usage:
  python evaluate_overlap.py
  python evaluate_overlap.py --series s1
  python evaluate_overlap.py --series s2 --verbose
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import VOCAB_SIZE
from models.transformer import AutoregressiveTransformer
from evaluate import (
    _load_seed_models,
    _eval_models,
    rule_following_acc,
    _fmt,
)


# ---------------------------------------------------------------------------
# Overlap experiment configs  (A, B match run_overlap_experiments.sh exactly)
# ---------------------------------------------------------------------------

NEVER_SEEN = [17, 19, 20, 21]   # fixed across all configs

SERIES = {
    's1': {
        'label': 'Series 1  |A|=6  |A∪B|=16',
        'configs': [
            ('s1_ov0', [0,1,2,3,4,5],
                       [6,7,8,9,10,11,12,13,14,15],
                       'ov=0'),
            ('s1_ov2', [0,1,2,3,6,7],
                       [6,7,8,9,10,11,12,13,14,15,16,18],
                       'ov=2'),
            ('s1_ov4', [0,1,6,7,8,9],
                       [6,7,8,9,10,11,12,13,14,15,16,18,22,23],
                       'ov=4'),
            ('s1_ov6', [6,7,8,9,10,11],
                       [6,7,8,9,10,11,12,13,14,15,16,18,22,23,0,1],
                       'ov=6'),
        ],
    },
    's2': {
        'label': 'Series 2  |A|=10  |A∪B|=20',
        'configs': [
            ('s2_ov0', [0,1,2,3,4,5,16,18,22,23],
                       [6,7,8,9,10,11,12,13,14,15],
                       'ov=0'),
            ('s2_ov2', [0,1,2,3,4,5,16,18,6,7],
                       [6,7,8,9,10,11,12,13,14,15,22,23],
                       'ov=2'),
            ('s2_ov4', [0,1,2,3,4,5,6,7,8,9],
                       [6,7,8,9,10,11,12,13,14,15,16,18,22,23],
                       'ov=4'),
            ('s2_ov6', [0,1,2,3,6,7,8,9,10,11],
                       [6,7,8,9,10,11,12,13,14,15,16,18,22,23,4,5],
                       'ov=6'),
        ],
    },
}

# Fixed category keys used in results dict — same across all configs.
# Starters for A\B / B\A / A∩B are computed per config; empty → skipped in
# per-config table but shown as '—' in the cross-overlap summary.
CAT_KEYS = ['A\\B', 'B\\A', 'A∩B', 'Never-seen', 'Overall']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ar_ckpt(path: str, args, device, label: str):
    from data.dataset import VOCAB_SIZE as VS
    model = AutoregressiveTransformer(
        vocab_size  = VS,
        max_seq_len = args.n_cycles * 4 + 10,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
    ).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f'  {label:<14}: loaded {path}')
    else:
        print(f'  {label:<14}: WARNING {path} not found — using random weights')
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-config evaluation — returns results keyed by CAT_KEYS
# ---------------------------------------------------------------------------

def eval_config(stem: str, A: list, B: list, ov_label: str, args, device):
    A_set   = set(A)
    B_set   = set(B)
    a_only  = sorted(A_set - B_set)
    b_only  = sorted(B_set - A_set)
    a_and_b = sorted(A_set & B_set)
    all_eval = sorted(set(a_only + b_only + a_and_b + NEVER_SEEN))

    # Map CAT_KEYS → starters (None means empty / not applicable)
    cat_starters = {
        'A\\B':       a_only  if a_only  else None,
        'B\\A':       b_only  if b_only  else None,
        'A∩B':        a_and_b if a_and_b else None,
        'Never-seen': NEVER_SEEN,
        'Overall':    all_eval,
    }

    ckpt_dir = args.ckpt_dir
    pt_path  = os.path.join(ckpt_dir, f'{stem}_pretrain.pt')
    ft_path  = os.path.join(ckpt_dir, f'{stem}_finetune.pt')

    pretrain_model = load_ar_ckpt(pt_path, args, device, 'Pretrain')
    finetune_model = load_ar_ckpt(ft_path, args, device, 'Finetune')

    # Adapters are built on top of the config's own pretrain checkpoint
    args.ar_ckpt = pt_path
    adapter_models = _load_seed_models(f'{stem}_adapter', args, device)
    n_seeds = len(adapter_models)
    multi   = n_seeds > 1
    print(f'  Adapter seeds : {n_seeds}')

    # Evaluate
    results = {}   # cat_key → (pt_mean, ft_mean, ad_mean, ad_std) or None
    for key in CAT_KEYS:
        starters = cat_starters[key]
        if starters is None:
            results[key] = None
            continue
        _, pt_mean, _ = _eval_models([pretrain_model], starters, args.n_cycles, args.prompt_len, device)
        _, ft_mean, _ = _eval_models([finetune_model], starters, args.n_cycles, args.prompt_len, device)
        _, ad_mean, ad_std = _eval_models(adapter_models, starters, args.n_cycles, args.prompt_len, device)
        results[key] = (pt_mean, ft_mean, ad_mean, ad_std)

    # Per-config table
    col_w    = 11
    col_w_ad = 16

    header = f"  {'Category':<14} {'Starters':<30}  {'Pretrain':>{col_w}}  {'Finetune':>{col_w}}  {'Adapter':>{col_w_ad}}"
    sep    = '  ' + '-' * (len(header) - 2)
    print()
    print(f'  A = {sorted(A_set)}')
    print(f'  B = {sorted(B_set)}')
    print(header)
    print(sep)

    for key in CAT_KEYS:
        starters = cat_starters[key]
        val      = results[key]
        if val is None:
            s_str = '—'
            row = f"  {key:<14} {s_str:<30}  {'—':>{col_w}}  {'—':>{col_w}}  {'—':>{col_w_ad}}"
        else:
            pt, ft, ad_mean, ad_std = val
            s_str = str(starters[:5]) + ('…' if len(starters) > 5 else '')
            ad_str = _fmt(ad_mean, ad_std, col_w_ad, multi)
            row = f"  {key:<14} {s_str:<30}  {pt:{col_w}.3f}  {ft:{col_w}.3f}  {ad_str}"
        print(row)

    print(sep)

    # Per-seed detail
    if args.verbose and multi:
        print(f'\n  Per-seed breakdown: {stem}_adapter')
        active_cats = [(k, cat_starters[k]) for k in CAT_KEYS if cat_starters[k] is not None]
        hdrs = '  '.join(f'{k[:12]:>12}' for k, _ in active_cats)
        print(f"    {'seed':>4}  {hdrs}")
        print(f"    {'-'*4}  {'-'*len(hdrs)}")
        for i, mdl in enumerate(adapter_models):
            accs = []
            for _, starters in active_cats:
                _, mean = rule_following_acc(mdl, starters, args.n_cycles, args.prompt_len, device)
                accs.append(mean)
            print(f"    {i:>4}  " + '  '.join(f'{a:>12.4f}' for a in accs))

    return results, cat_starters


# ---------------------------------------------------------------------------
# Cross-overlap summary
# ---------------------------------------------------------------------------

def cross_config_summary(series_results: list):
    """
    series_results: list of (ov_label, results_dict)
    Prints a compact table: rows=overlap levels, cols=category × {PT,FT,AD}
    """
    col = 7
    active_cats = [k for k in CAT_KEYS]
    sub   = '  '.join(f'{"PT":>{col}} {"FT":>{col}} {"AD":>{col}}' for _ in active_cats)
    heads = '  '.join(f'{k[:8]:>{col*3+4}}' for k in active_cats)

    print(f'\n  Cross-overlap summary  (PT=Pretrain, FT=Finetune, AD=Adapter)')
    print(f'  {"":>8}  {heads}')
    print(f'  {"overlap":>8}  {sub}')
    print(f'  {"-"*8}  {"-"*len(sub)}')

    for ov_label, res in series_results:
        row = f'  {ov_label:>8}  '
        for key in active_cats:
            val = res.get(key)
            if val is None:
                row += f'  {"—":>{col}} {"—":>{col}} {"—":>{col}}'
            else:
                pt, ft, ad_mean, _ = val
                row += f'  {pt:{col}.3f} {ft:{col}.3f} {ad_mean:{col}.3f}'
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Never-seen starters (fixed): {NEVER_SEEN}')

    selected_series = args.series if args.series else list(SERIES.keys())

    for series_key in selected_series:
        if series_key not in SERIES:
            print(f'Unknown series: {series_key}. Choose from: {list(SERIES.keys())}')
            continue

        series = SERIES[series_key]
        print(f'\n{"#"*80}')
        print(f'{series["label"]}')
        print(f'{"#"*80}')

        series_results = []
        for stem, A, B, ov_label in series['configs']:
            print(f'\n{"="*70}')
            print(f'  {ov_label}  stem={stem}')
            print(f'{"="*70}')
            res, _ = eval_config(stem, A, B, ov_label, args, device)
            series_results.append((ov_label, res))

        cross_config_summary(series_results)

    print('\n\n=== OVERLAP EVALUATION COMPLETE ===')


def get_args():
    p = argparse.ArgumentParser(description='Evaluate overlap experiment checkpoints')
    p.add_argument('--ckpt_dir',   type=str, default='checkpoints')
    p.add_argument('--ar_ckpt',    type=str, default='',
                   help='Overridden per config; kept for API compat with evaluate.py')
    p.add_argument('--ft_ckpt',    type=str, default='',
                   help='Overridden per config; kept for API compat with evaluate.py')
    p.add_argument('--d_model',    type=int, default=128)
    p.add_argument('--n_layers',   type=int, default=4)
    p.add_argument('--n_heads',    type=int, default=4)
    p.add_argument('--n_cycles',   type=int, default=8)
    p.add_argument('--prompt_len', type=int, default=1)
    p.add_argument('--force_fallback', action='store_true', default=True)
    p.add_argument('--verbose',    action='store_true',
                   help='Print per-seed breakdown for each config')
    p.add_argument('--series',     type=str, nargs='*', default=None,
                   choices=['s1', 's2'],
                   help='Which series to evaluate (default: both)')
    return p.parse_args()


if __name__ == '__main__':
    run(get_args())
