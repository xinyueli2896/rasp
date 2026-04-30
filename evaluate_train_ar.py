#!/usr/bin/env python3
"""
Evaluate all train_ar experiment checkpoints.

For each dataset config (2to16 / 0to16 / 0to16_plus), compares:
  Pretrain   : ar_pretrain_0to6
  Finetune   : ar_finetuned_<config>
  FrozenAR   : adapter_<config>_seed*
  TrainAR    : adapter_train_ar_<config>_seed*
  ScratchAR  : adapter_train_ar_scratch_<config>_seed*

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
from data.dataset import (
    EVAL_PRETRAIN_ONLY,
    EVAL_FINETUNE_ONLY,
    EVAL_BOTH,
    EVAL_NEITHER,
    VOCAB_SIZE,
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
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silent():
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        yield


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
    print(f'  [{cat_label.strip()}]')
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

def eval_config(config: str, ft_stem: str, pretrain_model, args, device):
    print(f'\n{"="*80}')
    print(f'CONFIG: {config}')
    print(f'{"="*80}')

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

    # Accuracy
    results = {}
    for lbl, mdls, _ in model_groups:
        results[lbl] = {}
        for cat_label, starters, _ in CATEGORIES:
            per, mean, std = _eval_models(mdls, starters, args.n_cycles, args.prompt_len, device)
            results[lbl][cat_label] = (per, mean, std)
        per, mean, std = _eval_models(mdls, ALL_STARTERS, args.n_cycles, args.prompt_len, device)
        results[lbl]['Overall'] = (per, mean, std)

    # ---- Rule-following accuracy table ----
    col_w    = 12
    col_w_ms = 17

    all_labels = [lbl for lbl, _, _ in model_groups]
    is_multi   = {lbl: multi for lbl, _, multi in model_groups}

    header = f"  {'Data Split':<20} {'Starters':<18}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        header += f" {lbl:>{w}}"
    sep = '  ' + '-' * (len(header) - 2)

    print()
    print(header)
    print(sep)
    for cat_label, _, starters_str in CATEGORIES:
        row = f"  {cat_label:<20} {starters_str:<18}"
        for lbl in all_labels:
            w = col_w_ms if is_multi[lbl] else col_w
            _, mean, std = results[lbl][cat_label]
            row += ' ' + _fmt(mean, std, w, is_multi[lbl])
        print(row)
    print(sep)
    row = f"  {'Overall':<20} {'all starters':<18}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        _, mean, std = results[lbl]['Overall']
        row += ' ' + _fmt(mean, std, w, is_multi[lbl])
    print(row)
    print(sep)

    # ---- Error distribution (on finetune-only starters — the hard partition) ----
    print()
    print('  Error distribution  (pred - expected) % 24  [fraction of tokens]')
    print('  Computed on Finetune-only starters (B\\A)')
    qual_models = {
        'Pretrain':  pretrain_model,
        'Finetune':  ft_model,
        'FrozenAR':  frozen_models[0],
        'TrainAR':   train_ar_models[0] if train_ar_models else frozen_models[0],
        'ScratchAR': scratch_models[0]  if scratch_models  else frozen_models[0],
    }
    print_error_dist(qual_models, EVAL_FINETUNE_ONLY, args.n_cycles, args.prompt_len, device)

    # ---- Generated sequence examples ----
    print()
    print('  Generated sequence examples')
    for cat_label, starters, starters_str in CATEGORIES:
        print_examples(qual_models, starters, cat_label.strip(),
                       args.n_cycles, args.prompt_len, device, n_show=2)

    # ---- Per-seed detail ----
    if args.verbose:
        for lbl, mdls, multi in model_groups:
            if not multi:
                continue
            print(f'  Per-seed: {lbl}')
            cat_hdrs = '  '.join(f'{c[0].strip()[:12]:>12}' for c in CATEGORIES) + f'  {"Overall":>12}'
            print(f"    {'seed':>4}  {cat_hdrs}")
            print(f"    {'-'*4}  {'-'*len(cat_hdrs)}")
            for i, mdl in enumerate(mdls):
                accs = []
                for _, starters, _ in CATEGORIES:
                    _, mean = rule_following_acc(mdl, starters, args.n_cycles, args.prompt_len, device)
                    accs.append(mean)
                _, ov = rule_following_acc(mdl, ALL_STARTERS, args.n_cycles, args.prompt_len, device)
                accs.append(ov)
                print(f"    {i:>4}  " + '  '.join(f'{a:>12.4f}' for a in accs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with _silent():
        pretrain_model = load_pretrain(args, device)

    selected = [(c, ft) for c, ft in ALL_CONFIGS if c in args.configs]
    if not selected:
        print(f'No matching configs. Available: {[c for c, _ in ALL_CONFIGS]}')
        return

    for config, ft_stem in selected:
        eval_config(config, ft_stem, pretrain_model, args, device)

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
