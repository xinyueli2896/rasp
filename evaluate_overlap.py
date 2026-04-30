#!/usr/bin/env python3
"""
Evaluate overlap experiment checkpoints.

For each config, loads:
  {stem}_pretrain.pt      — pretrained AR
  {stem}_finetune.pt      — finetuned AR
  {stem}_adapter_seed*.pt — adapter x5 seeds (frozen pretrained AR)

Eval categories per config:
  A\B         : starters seen only during pretrain     (varies per config)
  B\\A        : starters seen only during finetune     (varies per config)
  A∩B         : starters seen in both                  (varies per config)
  Never-seen  : {17,19,20,21}                          (fixed)
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
import contextlib
import collections

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
    make_sequence,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

NEVER_SEEN = [17, 19, 20, 21]

SERIES = {
    's1': {
        'label': 'Series 1  B={6..15} fixed  |A∪B|=20 fixed  A grows with overlap',
        'configs': [
            # B = {6..15} fixed. A\B = {0,1,2,3,4,5,16,18,22,23} fixed.
            # A∩B grows → |A| grows from 10→12→14→16; |A∪B|=20 always.
            # A always covers tokens 0-4; B always covers tokens 17,19-22.
            ('s1_ov0', [0,1,2,3,4,5,16,18,22,23],
                       [6,7,8,9,10,11,12,13,14,15],            'ov=0'),
            ('s1_ov2', [0,1,2,3,4,5,16,18,22,23,6,7],
                       [6,7,8,9,10,11,12,13,14,15],            'ov=2'),
            ('s1_ov4', [0,1,2,3,4,5,16,18,22,23,6,7,8,9],
                       [6,7,8,9,10,11,12,13,14,15],            'ov=4'),
            ('s1_ov6', [0,1,2,3,4,5,16,18,22,23,6,7,8,9,10,11],
                       [6,7,8,9,10,11,12,13,14,15],            'ov=6'),
        ],
    },
    's2': {
        'label': 'Series 2  |A|=10  |A∪B|=20',
        'configs': [
            ('s2_ov0', [0,1,2,3,4,5,16,18,22,23],
                       [6,7,8,9,10,11,12,13,14,15],                'ov=0'),
            ('s2_ov2', [0,1,2,3,4,5,16,18,6,7],
                       [6,7,8,9,10,11,12,13,14,15,22,23],          'ov=2'),
            ('s2_ov4', [0,1,2,3,4,5,6,7,8,9],
                       [6,7,8,9,10,11,12,13,14,15,16,18,22,23],    'ov=4'),
            ('s2_ov6', [0,1,2,3,6,7,8,9,10,11],
                       [6,7,8,9,10,11,12,13,14,15,16,18,22,23,4,5],'ov=6'),
        ],
    },
}

CAT_KEYS = ['A\\B', 'B\\A', 'A∩B', 'Never-seen', 'Overall']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silent():
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        yield


def _load_ar(path: str, args, device):
    model = AutoregressiveTransformer(
        vocab_size  = VOCAB_SIZE,
        max_seq_len = args.n_cycles * 4 + 10,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
    ).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Error distribution
# ---------------------------------------------------------------------------

@torch.no_grad()
def error_distribution(model, starters, n_cycles, n_prompt, device):
    """Return Counter of (pred - expected) % VOCAB_SIZE across all generated tokens."""
    gen_len = n_cycles * 4 - n_prompt
    counter = collections.Counter()
    for x in starters:
        seq      = make_sequence(x, n_cycles)
        prompt   = torch.tensor([seq[:n_prompt]], dtype=torch.long, device=device)
        expected = seq[n_prompt:]
        generated = model.generate(prompt, n_new=gen_len)[0, n_prompt:].cpu().tolist()
        for g, e in zip(generated, expected):
            counter[(g - e) % VOCAB_SIZE] += 1
    return counter


def print_error_dist(models_dict: dict, starters: list, n_cycles: int, n_prompt: int, device):
    """Print error distribution table for each model in models_dict."""
    if not starters:
        return
    all_counters = {}
    for label, model in models_dict.items():
        all_counters[label] = error_distribution(model, starters, n_cycles, n_prompt, device)

    # Collect top errors across all models
    total = collections.Counter()
    for c in all_counters.values():
        total.update(c)
    top_errors = [err for err, _ in total.most_common(8)]
    top_errors = sorted(top_errors)

    col = 8
    labels = list(models_dict.keys())
    header = f"  {'error':>6}  " + '  '.join(f'{l:>{col}}' for l in labels)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for err in top_errors:
        total_tokens = sum(all_counters[l].total() for l in labels) // len(labels)
        row = f"  {err:>6}  "
        for l in labels:
            cnt = all_counters[l].get(err, 0)
            tot = all_counters[l].total()
            pct = cnt / tot if tot > 0 else 0.0
            row += f'  {pct:{col}.3f}'
        marker = ' ← correct' if err == 0 else ''
        print(row + marker)


# ---------------------------------------------------------------------------
# Generated sequence examples
# ---------------------------------------------------------------------------

@torch.no_grad()
def print_examples(models_dict: dict, starters: list, cat_label: str,
                   n_cycles: int, n_prompt: int, device, n_show: int = 3):
    if not starters:
        return
    show_starters = starters[:n_show]
    show_len      = n_cycles * 4 - n_prompt

    print(f'  [{cat_label}]')
    for x in show_starters:
        seq      = make_sequence(x, n_cycles)
        prompt   = seq[:n_prompt]
        expected = seq[n_prompt: n_prompt + show_len]
        inp      = torch.tensor([prompt], dtype=torch.long, device=device)

        print(f'    x={x:<3}  prompt={prompt}  expected={expected}')
        for label, model in models_dict.items():
            gen  = model.generate(inp, n_new=show_len)[0, n_prompt:].cpu().tolist()
            mark = '✓' if gen == expected else '✗'
            print(f'    {label:<14}: {gen}  {mark}')
        print()


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def eval_config(stem: str, A: list, B: list, ov_label: str, args, device):
    A_set   = set(A)
    B_set   = set(B)
    a_only  = sorted(A_set - B_set)
    b_only  = sorted(B_set - A_set)
    a_and_b = sorted(A_set & B_set)
    all_eval = sorted(set(a_only + b_only + a_and_b + NEVER_SEEN))

    cat_starters = {
        'A\\B':       a_only  if a_only  else None,
        'B\\A':       b_only  if b_only  else None,
        'A∩B':        a_and_b if a_and_b else None,
        'Never-seen': NEVER_SEEN,
        'Overall':    all_eval,
    }

    pt_path = os.path.join(args.ckpt_dir, f'{stem}_pretrain.pt')
    ft_path = os.path.join(args.ckpt_dir, f'{stem}_finetune.pt')

    with _silent():
        pretrain_model = _load_ar(pt_path, args, device)
        finetune_model = _load_ar(ft_path, args, device)
        args.ar_ckpt   = pt_path
        adapter_models = _load_seed_models(f'{stem}_adapter', args, device)

    n_seeds = len(adapter_models)
    multi   = n_seeds > 1
    # Use seed-0 adapter for qualitative outputs
    adapter0 = adapter_models[0]

    # Evaluate accuracy
    results = {}
    for key in CAT_KEYS:
        starters = cat_starters[key]
        if starters is None:
            results[key] = None
            continue
        _, pt_mean, _ = _eval_models([pretrain_model], starters, args.n_cycles, args.prompt_len, device)
        _, ft_mean, _ = _eval_models([finetune_model], starters, args.n_cycles, args.prompt_len, device)
        _, ad_mean, ad_std = _eval_models(adapter_models, starters, args.n_cycles, args.prompt_len, device)
        results[key] = (pt_mean, ft_mean, ad_mean, ad_std)

    # ---- Rule-following accuracy table ----
    col_w    = 11
    col_w_ad = 16
    print(f'  A = {sorted(A_set)}')
    print(f'  B = {sorted(B_set)}')
    print(f'  Adapter seeds: {n_seeds}')
    print()

    header = f"  {'Category':<14} {'Starters':<30}  {'Pretrain':>{col_w}}  {'Finetune':>{col_w}}  {'Adapter (mean±std)':>{col_w_ad}}"
    sep    = '  ' + '-' * (len(header) - 2)
    print(header)
    print(sep)
    for key in CAT_KEYS:
        starters = cat_starters[key]
        val      = results[key]
        if val is None:
            print(f"  {key:<14} {'—':<30}  {'—':>{col_w}}  {'—':>{col_w}}  {'—':>{col_w_ad}}")
        else:
            pt, ft, ad_mean, ad_std = val
            s_str  = str(starters[:5]) + ('…' if len(starters) > 5 else '')
            ad_str = _fmt(ad_mean, ad_std, col_w_ad, multi)
            print(f"  {key:<14} {s_str:<30}  {pt:{col_w}.3f}  {ft:{col_w}.3f}  {ad_str}")
    print(sep)

    # ---- Error distribution ----
    print()
    print('  Error distribution  (pred - expected) % 24  [fraction of tokens]')
    ref_starters = (cat_starters['B\\A'] or cat_starters['Overall'])[:8]
    print_error_dist(
        {'Pretrain': pretrain_model, 'Finetune': finetune_model, 'Adapter': adapter0},
        ref_starters, args.n_cycles, args.prompt_len, device,
    )

    # ---- Generated sequence examples ----
    print()
    print('  Generated sequence examples')
    for key in ['A\\B', 'B\\A', 'A∩B', 'Never-seen']:
        starters = cat_starters[key]
        if starters:
            print_examples(
                {'Pretrain': pretrain_model, 'Finetune': finetune_model, 'Adapter': adapter0},
                starters, key, args.n_cycles, args.prompt_len, device, n_show=2,
            )

    # ---- Per-seed detail ----
    if args.verbose and multi:
        print('  Per-seed breakdown')
        active = [(k, cat_starters[k]) for k in CAT_KEYS if cat_starters[k] is not None]
        hdrs   = '  '.join(f'{k[:12]:>12}' for k, _ in active)
        print(f"    {'seed':>4}  {hdrs}")
        print(f"    {'-'*4}  {'-'*len(hdrs)}")
        for i, mdl in enumerate(adapter_models):
            accs = []
            for _, starters in active:
                _, mean = rule_following_acc(mdl, starters, args.n_cycles, args.prompt_len, device)
                accs.append(mean)
            print(f"    {i:>4}  " + '  '.join(f'{a:>12.4f}' for a in accs))

    return results, cat_starters


# ---------------------------------------------------------------------------
# Cross-overlap summary
# ---------------------------------------------------------------------------

def cross_config_summary(series_results: list):
    col = 7
    sub   = '  '.join(f'{"PT":>{col}} {"FT":>{col}} {"AD":>{col}}' for _ in CAT_KEYS)
    heads = '  '.join(f'{k[:8]:>{col*3+4}}' for k in CAT_KEYS)

    print()
    print('  Cross-overlap accuracy summary  (PT=Pretrain  FT=Finetune  AD=Adapter)')
    print(f'  {"":>8}  {heads}')
    print(f'  {"overlap":>8}  {sub}')
    print(f'  {"-"*8}  {"-"*len(sub)}')
    for ov_label, res in series_results:
        row = f'  {ov_label:>8}  '
        for key in CAT_KEYS:
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

    selected = args.series if args.series else list(SERIES.keys())

    for series_key in selected:
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
    p.add_argument('--ar_ckpt',    type=str, default='')
    p.add_argument('--ft_ckpt',    type=str, default='')
    p.add_argument('--d_model',    type=int, default=128)
    p.add_argument('--n_layers',   type=int, default=4)
    p.add_argument('--n_heads',    type=int, default=4)
    p.add_argument('--n_cycles',   type=int, default=8)
    p.add_argument('--prompt_len', type=int, default=1)
    p.add_argument('--force_fallback', action='store_true', default=True)
    p.add_argument('--verbose',    action='store_true')
    p.add_argument('--series',     type=str, nargs='*', default=None,
                   choices=['s1', 's2'])
    return p.parse_args()


if __name__ == '__main__':
    run(get_args())
