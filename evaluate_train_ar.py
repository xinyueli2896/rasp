#!/usr/bin/env python3
"""
Evaluate all train_ar experiment checkpoints.

For each dataset config (2to16 / 0to16 / 0to16_plus), compares:
  Pretrain   : ar_pretrain_0to6
  Finetune   : ar_finetuned_<config>
  FrozenAR   : adapter_<config>_seed*
  TrainAR    : adapter_train_ar_<config>_seed*
  ScratchAR  : adapter_train_ar_scratch_<config>_seed*

Eval categories are computed dynamically from A (pretrain) and B (adapter)
per config, so A\\B / B\\A / A∩B show correct starters for each config.
Never-seen = {17,19,20,21} is fixed across all configs.

Usage:
  python evaluate_train_ar.py
  python evaluate_train_ar.py --configs 2to16
  python evaluate_train_ar.py --verbose
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

from evaluate import (
    load_pretrain,
    load_finetune,
    _load_seed_models,
    _eval_models,
    rule_following_acc,
    _fmt,
    make_sequence,
)
from data.dataset import VOCAB_SIZE


NEVER_SEEN = [17, 19, 20, 21]   # fixed across all configs

# (config_name, ft_ckpt_stem, A=pretrain_starters, B=adapter_train_starters)
ALL_CONFIGS = [
    ('2to16',      'ar_finetuned_2to16',      list(range(6)),                       list(range(2, 17))),
    ('0to16',      'ar_finetuned_0to16',       list(range(6)),                       list(range(0, 17))),
    ('0to16_plus', 'ar_finetuned_0to16_plus',  list(range(6)),                       list(range(0, 17)) + [18, 22, 23]),
]

CAT_KEYS = ['A\\B', 'B\\A', 'A∩B', 'Never-seen', 'Overall']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silent():
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        yield


def _cat_starters(A: list, B: list):
    A_set, B_set = set(A), set(B)
    a_only  = sorted(A_set - B_set)
    b_only  = sorted(B_set - A_set)
    a_and_b = sorted(A_set & B_set)
    overall = sorted(set(a_only + b_only + a_and_b + NEVER_SEEN))
    return {
        'A\\B':       a_only  if a_only  else None,
        'B\\A':       b_only  if b_only  else None,
        'A∩B':        a_and_b if a_and_b else None,
        'Never-seen': NEVER_SEEN,
        'Overall':    overall,
    }


# ---------------------------------------------------------------------------
# Error distribution
# ---------------------------------------------------------------------------

@torch.no_grad()
def error_distribution(model, starters, n_cycles, n_prompt, device):
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
    if not starters:
        return
    all_counters = {l: error_distribution(m, starters, n_cycles, n_prompt, device)
                    for l, m in models_dict.items()}
    total = collections.Counter()
    for c in all_counters.values():
        total.update(c)
    top_errors = sorted(err for err, _ in total.most_common(8))

    col    = 10
    labels = list(models_dict.keys())
    print('  ' + f"{'error':>6}  " + '  '.join(f'{l:>{col}}' for l in labels))
    print('  ' + '-' * (6 + 2 + (col + 2) * len(labels)))
    for err in top_errors:
        row = f"  {err:>6}  "
        for l in labels:
            tot = all_counters[l].total()
            pct = all_counters[l].get(err, 0) / tot if tot > 0 else 0.0
            row += f'  {pct:{col}.3f}'
        print(row + (' ← correct' if err == 0 else ''))


# ---------------------------------------------------------------------------
# Generated sequence examples
# ---------------------------------------------------------------------------

@torch.no_grad()
def print_examples(models_dict: dict, starters: list, cat_label: str,
                   n_cycles: int, n_prompt: int, device, n_show: int = 2):
    if not starters:
        return
    show_len = n_cycles * 4 - n_prompt
    print(f'  [{cat_label}]')
    for x in starters[:n_show]:
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

def eval_config(config: str, ft_stem: str, A: list, B: list,
                pretrain_model, args, device):
    print(f'\n{"="*80}')
    print(f'CONFIG: {config}')
    print(f'  A (pretrain)         = {sorted(A)}')
    print(f'  B (adapter/finetune) = {sorted(B)}')
    print(f'{"="*80}')

    cat_s = _cat_starters(A, B)

    args.ft_ckpt = os.path.join(args.ckpt_dir, f'{ft_stem}.pt')
    with _silent():
        ft_model        = load_finetune(args, device)
        frozen_models   = _load_seed_models(f'adapter_{config}',                  args, device)
        train_ar_models = _load_seed_models(f'adapter_train_ar_{config}',         args, device)
        scratch_models  = _load_seed_models(f'adapter_train_ar_scratch_{config}', args, device)

    model_groups = [
        ('Pretrain',  [pretrain_model], False),
        ('Finetune',  [ft_model],       False),
        ('FrozenAR',  frozen_models,    len(frozen_models) > 1),
        ('TrainAR',   train_ar_models,  len(train_ar_models) > 1),
        ('ScratchAR', scratch_models,   len(scratch_models) > 1),
    ]

    print(f'  Seeds — FrozenAR:{len(frozen_models)}  TrainAR:{len(train_ar_models)}  ScratchAR:{len(scratch_models)}')

    # Evaluate accuracy per category
    results = {}
    for lbl, mdls, _ in model_groups:
        results[lbl] = {}
        for key in CAT_KEYS:
            starters = cat_s[key]
            if starters is None:
                results[lbl][key] = None
                continue
            _, mean, std = _eval_models(mdls, starters, args.n_cycles, args.prompt_len, device)
            results[lbl][key] = (mean, std)

    # ---- Rule-following accuracy table ----
    col_w    = 12
    col_w_ms = 17

    all_labels = [lbl for lbl, _, _ in model_groups]
    is_multi   = {lbl: multi for lbl, _, multi in model_groups}

    header = f"  {'Category':<16} {'Starters':<28}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        header += f" {lbl:>{w}}"
    sep = '  ' + '-' * (len(header) - 2)

    print()
    print(header)
    print(sep)
    for key in CAT_KEYS:
        starters = cat_s[key]
        s_str    = ('—' if starters is None
                    else str(starters[:5]) + ('…' if len(starters) > 5 else ''))
        row = f"  {key:<16} {s_str:<28}"
        for lbl in all_labels:
            w   = col_w_ms if is_multi[lbl] else col_w
            val = results[lbl][key]
            if val is None:
                row += f" {'—':>{w}}"
            else:
                mean, std = val
                row += ' ' + _fmt(mean, std, w, is_multi[lbl])
        print(row)
    print(sep)

    # ---- Error distribution (on B\A — the "hard" generalization partition) ----
    err_starters = cat_s['B\\A'] or cat_s['Overall']
    print()
    print(f'  Error distribution  (pred - expected) % 24  [fraction of tokens]')
    print(f'  Computed on: B\\A = {err_starters[:8]}{"…" if len(err_starters) > 8 else ""}')
    qual_models = {lbl: mdls[0] for lbl, mdls, _ in model_groups}
    print_error_dist(qual_models, err_starters[:10], args.n_cycles, args.prompt_len, device)

    # ---- Generated sequence examples ----
    print()
    print('  Generated sequence examples')
    for key in CAT_KEYS[:-1]:   # skip Overall
        print_examples(qual_models, cat_s[key] or [], key,
                       args.n_cycles, args.prompt_len, device, n_show=2)

    # ---- Per-seed detail ----
    if args.verbose:
        for lbl, mdls, multi in model_groups:
            if not multi:
                continue
            print(f'  Per-seed: {lbl}')
            active = [(k, cat_s[k]) for k in CAT_KEYS if cat_s[k] is not None]
            hdrs   = '  '.join(f'{k[:12]:>12}' for k, _ in active)
            print(f"    {'seed':>4}  {hdrs}")
            print(f"    {'-'*4}  {'-'*len(hdrs)}")
            for i, mdl in enumerate(mdls):
                accs = []
                for _, starters in active:
                    _, mean = rule_following_acc(mdl, starters, args.n_cycles, args.prompt_len, device)
                    accs.append(mean)
                print(f"    {i:>4}  " + '  '.join(f'{a:>12.4f}' for a in accs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with _silent():
        pretrain_model = load_pretrain(args, device)

    selected = [(c, ft, A, B) for c, ft, A, B in ALL_CONFIGS if c in args.configs]
    if not selected:
        print(f'No matching configs. Available: {[c for c, *_ in ALL_CONFIGS]}')
        return

    for config, ft_stem, A, B in selected:
        eval_config(config, ft_stem, A, B, pretrain_model, args, device)

    print('\n\n=== EVALUATION COMPLETE ===')


def get_args():
    p = argparse.ArgumentParser(description='Evaluate train_ar experiment checkpoints')
    p.add_argument('--ar_ckpt',    type=str, default='checkpoints/ar_pretrain_0to6.pt')
    p.add_argument('--ft_ckpt',    type=str, default='checkpoints/ar_finetuned_2to16.pt')
    p.add_argument('--ckpt_dir',   type=str, default='checkpoints')
    p.add_argument('--d_model',    type=int, default=128)
    p.add_argument('--n_layers',   type=int, default=4)
    p.add_argument('--n_heads',    type=int, default=4)
    p.add_argument('--n_cycles',   type=int, default=8)
    p.add_argument('--prompt_len', type=int, default=1)
    p.add_argument('--verbose',    action='store_true')
    p.add_argument('--force_fallback', action='store_true', default=True)
    p.add_argument('--configs',    type=str, nargs='*',
                   default=['2to16', '0to16', '0to16_plus'])
    return p.parse_args()


if __name__ == '__main__':
    run(get_args())
