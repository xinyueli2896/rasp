"""
Aggregate CP bass rule-following accuracy across seeds into ONE mean±std table,
in the same shape as the integer experiment's evaluate.py output.

evaluate_cp_yinyang.py evaluates a single --adapter_ckpt and has no multi-seed
aggregation, so a 5-seed × 5-mode sweep otherwise means 25 separate tables to
average by hand.  This script loops the per-seed run directories for each mode
and prints one table.

It does not reimplement the metric: rule_following_acc and load_model are
imported from evaluate_cp_bass (bare base models) and evaluate_cp_yinyang
(adapter models), and the key partitions come from evaluate_cp_bass.

Architecture flags are DERIVED FROM THE STEM NAME rather than restated on the
command line, following midi_adapter/probe_run.sh: an adapter rebuilt at the
wrong shape loads none of its weights and still prints a plausible table.  The
derived flags are printed per stem so the choice is auditable.

Usage
-----
  python -m midi_adapter.cp_results_table \\
      --base_ckpt_dir ckpt --base_run cp_bass_pre_A --ft_run cp_bass_ft_B \\
      --ckpt_dir checkpoints_cp \\
      --adapter_stems cp_yy_frozen cp_yy_trainar cp_yy_lora cp_yy_scratch cp_yy_enc \\
      --approach bass --model_size 1 --adapter_rank 256 --n_skip 1

Model selection
---------------
--select val_loss (default) takes the lowest-val_loss checkpoint of each run,
which is the rule probe_run.sh and the CP runners use.  --select unseen_acc
takes the highest unseen_bass_acc instead, which is what the integer adapter
runs did (_run_training checkpoints on best test_acc(unseen)).  Use
unseen_acc for a like-for-like comparison against the integer table, and keep
in mind that it selects on the quantity being reported.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.evaluate_cp_bass import (
    CATEGORIES, ALL_SEEN, UNSEEN,
    load_model         as load_base_model,
    rule_following_acc as base_rule_following_acc,
)
from midi_adapter.evaluate_cp_yinyang import (
    load_model         as load_adapter_model,
    rule_following_acc as adapter_rule_following_acc,
)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_SERIES = {
    # name            filename marker      metric field        better
    'val_loss':   ('.by_val_loss.',   'val_loss',        min),
    'unseen_acc': ('.by_unseen_acc.', 'unseen_bass_acc', max),
}


def _pick(paths: list[str], field: str, better) -> str | None:
    """Return the path whose `field=<number>` in the filename is best."""
    # The number must not swallow the '.' before '.ckpt': a greedy [0-9.]+ on
    # 'val_loss=0.41000.ckpt' captures '0.41000.', which float() rejects, and
    # every candidate is then silently discarded.
    pattern = rf'{re.escape(field)}=(-?\d+(?:\.\d+)?)'
    scored = []
    for p in paths:
        m = re.search(pattern, os.path.basename(p))
        if m:
            scored.append((float(m.group(1)), p))
    if not scored:
        return None
    return better(scored, key=lambda t: t[0])[1]


def best_adapter_ckpt(run_dir: str, select: str) -> str | None:
    """Best checkpoint of one adapter run directory, by the chosen series."""
    marker, field, better = _SERIES[select]
    paths = [p for p in glob.glob(os.path.join(run_dir, '*.ckpt')) if marker in p]
    return _pick(paths, field, better)


def best_base_ckpt(run_dir: str) -> str | None:
    """Lowest-val_loss checkpoint of a pretrain / finetune run directory.

    Those runners use a single series named `<run>.{epoch}.{val_loss}.ckpt`
    with no `.by_*.` marker, so anything carrying one is excluded, as is
    last.ckpt (no metric in its name to rank by).
    """
    paths = [p for p in glob.glob(os.path.join(run_dir, '*.ckpt'))
             if '.by_' not in os.path.basename(p)
             and os.path.basename(p) != 'last.ckpt']
    return _pick(paths, 'val_loss', min)


def seed_run_dirs(ckpt_dir: str, stem: str) -> list[str]:
    """Per-seed run directories for one mode, ordered by seed number."""
    dirs = [d for d in glob.glob(os.path.join(ckpt_dir, f'{stem}_seed*'))
            if os.path.isdir(d)]

    def _seed(d):
        m = re.search(r'_seed(\d+)$', os.path.basename(d))
        return int(m.group(1)) if m else 10**6

    return sorted(dirs, key=_seed)


# ---------------------------------------------------------------------------
# Per-stem architecture flags, derived from the stem name
# ---------------------------------------------------------------------------

def derive_flags(stem: str, args) -> dict:
    """Architecture kwargs for load_adapter_model, read off the stem name.

    Only the flags that change module SHAPES are derived here; --unfreeze_base
    and --pretrain_data change what trains, not what is built, so a frozen,
    a trainable and a from-scratch run all load identically.
    """
    low = stem.lower()
    flags = {
        'bidirectional':    'bidir' in low,
        'encoder_injected': 'enc' in low,
        'lora_rank':        args.lora_rank if 'lora' in low else 0,
    }
    if flags['encoder_injected']:
        flags['encoder_type'] = args.encoder_type
    return flags


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_one(model, is_adapter: bool, keys: list[int], args, device) -> float:
    rfa = adapter_rule_following_acc if is_adapter else base_rule_following_acc
    per_key, _ = rfa(model, keys, args.n_gen_beats, args.n_trials,
                     device, args.temperature, args.n_prompt_beats)
    return float(np.mean([per_key[k] for k in keys]))


def eval_models(models, is_adapter: bool, keys, args, device) -> tuple[float, float]:
    accs = [eval_one(m, is_adapter, keys, args, device) for m in models]
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _cell(mean: float | None, std: float, multi: bool, width: int) -> str:
    if mean is None:
        return 'N/A'.rjust(width)
    s = f'{mean:.3f}±{std:.3f}' if multi else f'{mean:.3f}'
    return s.rjust(width)


def print_table(results: dict, labels: list[str], n_seeds: dict, rows: list) -> None:
    def width(lbl):
        data = max(len(_cell(0.0, 0.0, n_seeds[lbl] > 1, 0)), len(lbl))
        return max(data, 8)

    header = f"{'Data Split':<20} {'Keys':<16}"
    for lbl in labels:
        header += f' {lbl:>{width(lbl)}}'
    sep = '-' * len(header)

    print()
    print('=' * len(header))
    print('CP BASS RULE-FOLLOWING ACCURACY')
    print('=' * len(header))
    print(sep)
    print(header)
    print(sep)
    for row_label, keys_str, keys in rows:
        line = f'{row_label:<20} {keys_str:<16}'
        for lbl in labels:
            mean, std = results[lbl].get(row_label, (None, 0.0))
            line += f' {_cell(mean, std, n_seeds[lbl] > 1, width(lbl))}'
        print(line)
        if row_label == rows[-2][0]:
            print(sep)
    print(sep)

    seeds_line = f"{'seeds':<20} {'':<16}"
    for lbl in labels:
        seeds_line += f' {str(n_seeds[lbl]):>{width(lbl)}}'
    print(seeds_line)
    print('=' * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))

    all_keys = ALL_SEEN + UNSEEN
    rows = [(label, keys_str, keys) for label, keys, keys_str in CATEGORIES]
    rows.append(('Overall', 'all keys', all_keys))

    print('=' * 60)
    print('Loading models')
    print('=' * 60)
    print(f'  device       : {device}')
    print(f'  select       : {args.select}')
    print(f'  n_gen_beats  : {args.n_gen_beats}   n_prompt_beats: {args.n_prompt_beats}'
          f'   n_trials: {args.n_trials}')

    # --- base checkpoints (modes 1 and 2) ---------------------------------
    base_ckpt = args.base_ckpt
    if not base_ckpt and args.base_run:
        base_ckpt = best_base_ckpt(os.path.join(args.base_ckpt_dir, args.base_run))
    ft_ckpt = args.ft_ckpt
    if not ft_ckpt and args.ft_run:
        ft_ckpt = best_base_ckpt(os.path.join(args.base_ckpt_dir, args.ft_run))

    single = []
    for label, ckpt in (('Pretrain', base_ckpt), ('Finetune', ft_ckpt)):
        if not ckpt:
            print(f'  {label:<12}: no checkpoint resolved — column omitted')
            continue
        if not os.path.exists(ckpt):
            # Same rule as evaluate_cp_yinyang: a missing checkpoint is a hard
            # error, not a warning, or the table reports random weights.
            raise SystemExit(f'{label} checkpoint does not exist: {ckpt!r}')
        print(f'  {label:<12}: {ckpt}')
        single.append((label, [load_base_model(ckpt, args.model_size, device)]))

    # --- adapter checkpoints (modes 3-7) ----------------------------------
    adapters = []
    for stem in args.adapter_stems:
        run_dirs = seed_run_dirs(args.ckpt_dir, stem)
        if not run_dirs:
            raise SystemExit(
                f'no run directories matching {os.path.join(args.ckpt_dir, stem)}_seed*\n'
                f'  Refusing to report a column with no checkpoints behind it.')
        flags = derive_flags(stem, args)
        print(f'\n  {stem}: {len(run_dirs)} seed dir(s)')
        print(f'    derived flags: {flags}')

        models = []
        for d in run_dirs:
            ckpt = best_adapter_ckpt(d, args.select)
            if ckpt is None:
                print(f'    {os.path.basename(d):<28} no .{args.select} checkpoint — SKIPPED')
                continue
            print(f'    {os.path.basename(d):<28} {os.path.basename(ckpt)}')
            models.append(load_adapter_model(
                base_ckpt, ckpt,
                args.model_size, args.adapter_rank, args.n_skip,
                flags['bidirectional'], flags['encoder_injected'],
                encoder_type = flags.get('encoder_type', 'embedding'),
                rule_mode    = args.rule_mode,
                approach     = args.approach,
                lora_rank    = flags['lora_rank'],
                device       = device,
            ))
        if not models:
            raise SystemExit(f'stem {stem!r}: no loadable checkpoints.')
        adapters.append((stem, models))

    # --- evaluate ---------------------------------------------------------
    print()
    print('=' * 60)
    print('Evaluating')
    print('=' * 60)
    adapter_labels = {label for label, _ in adapters}
    results: dict = {}
    n_seeds: dict = {}
    for label, models in single + adapters:
        results[label] = {}
        n_seeds[label] = len(models)
        print(f'  {label} ({len(models)} seed(s)) ...', end='  ', flush=True)
        for row_label, _, keys in rows:
            results[label][row_label] = eval_models(
                models, label in adapter_labels, keys, args, device)
        print('  '.join(f'{results[label][r][0]:.3f}' for r, _, _ in rows))

    labels = [l for l, _ in single] + [l for l, _ in adapters]
    print_table(results, labels, n_seeds, rows)

    if args.select == 'val_loss':
        print('\nNote: checkpoints selected by lowest val_loss. The integer adapter')
        print('runs selected on best unseen accuracy instead, so pass')
        print('--select unseen_acc for a like-for-like comparison with that table.')


def get_args():
    p = argparse.ArgumentParser(
        description='Aggregate CP bass rule-following accuracy across seeds into '
                    'one mean±std table.')
    p.add_argument('--ckpt_dir',       type=str, default='checkpoints_cp',
                   help='Directory holding the per-seed ADAPTER run directories.')
    p.add_argument('--base_ckpt_dir',  type=str, default='ckpt',
                   help='Directory holding the pretrain / finetune run directories.')
    p.add_argument('--base_run',       type=str, default=None,
                   help='Pretrain run name; its best-val_loss ckpt becomes the '
                        'Pretrain column and the base weights for adapter-only .pt files.')
    p.add_argument('--ft_run',         type=str, default=None,
                   help='Finetune-without-adapter run name (the Finetune column).')
    p.add_argument('--base_ckpt',      type=str, default=None,
                   help='Explicit pretrain checkpoint path, overriding --base_run.')
    p.add_argument('--ft_ckpt',        type=str, default=None,
                   help='Explicit finetune checkpoint path, overriding --ft_run.')
    p.add_argument('--adapter_stems',  type=str, nargs='+', default=[],
                   help='Run-name stems WITHOUT the _seedN suffix, one per mode. '
                        'Architecture flags are derived from each stem: "lora" '
                        '-> --lora_rank, "enc" -> --encoder_injected, "bidir" '
                        '-> --bidirectional.')
    p.add_argument('--select',         type=str, default='val_loss',
                   choices=['val_loss', 'unseen_acc'],
                   help='Which checkpoint series to take from each adapter run.')
    p.add_argument('--model_size',     type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--adapter_rank',   type=int, default=256)
    p.add_argument('--n_skip',         type=int, default=1,
                   help='MUST match training. A mismatch builds the adapter at '
                        'the wrong depth and its weights silently do not load.')
    p.add_argument('--lora_rank',      type=int, default=16,
                   help='Rank used for stems whose name contains "lora".')
    p.add_argument('--encoder_type',   type=str, default='embedding',
                   choices=['embedding', 'token_mlp'],
                   help='Encoder type for stems whose name contains "enc".')
    p.add_argument('--rule_mode',      type=str, default='current')
    p.add_argument('--approach',       type=str, default='bass',
                   choices=['bass', 'chord'])
    p.add_argument('--n_gen_beats',    type=int, default=64)
    p.add_argument('--n_prompt_beats', type=int, default=2)
    p.add_argument('--n_trials',       type=int, default=1)
    p.add_argument('--temperature',    type=float, default=0.0)
    p.add_argument('--device',         type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    run(get_args())
