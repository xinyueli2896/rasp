"""
Evaluate rule-following accuracy of the CP bass transformer.

For each key, generate from a 1-beat prompt (the starter note x only) and check
whether the predicted pitch at every position follows the cadence rule:
    expected[pos] = (key + OFFSETS[pos % 4]) % 12
    OFFSETS = [0, 5, 7, 0]   (I – IV – V – I)

Mirrors evaluate.py from the integer RASP experiment.

Usage
-----
  python -m midi_adapter.evaluate_cp_bass \\
      --base_ckpt ckpt/cp_bass_size1_batch8/last.ckpt

  # stochastic (multiple trials)
  python -m midi_adapter.evaluate_cp_bass \\
      --base_ckpt ckpt/cp_bass_size1_batch8/last.ckpt \\
      --temperature 0.8 --n_trials 8 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS
from midi_adapter.infer_cp_bass import _sample, _prompt_from_key

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Key partitions (mirror of EVAL_PRETRAIN_ONLY / EVAL_FINETUNE_ONLY / ... in evaluate.py)
PRETRAIN_KEYS = [0, 2, 4, 5, 7, 9, 11]     # seen during CP pretraining
FINETUNE_NEW  = [1, 3, 10]                  # added in fine-tune, not seen in pretrain
ALL_SEEN      = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11]
UNSEEN        = [6, 8]                      # never seen

CATEGORIES = [
    ('Pretrain keys', PRETRAIN_KEYS, '{0,2,4,5,7,9,11}'),
    ('Finetune-new',  FINETUNE_NEW,  '{1,3,10}'),
    ('All seen',      ALL_SEEN,      '{0..5,7,9..11}'),
    ('Unseen',        UNSEEN,        '{6,8}'),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_pc(key: int, pos: int) -> int:
    """Expected pitch class at absolute beat position pos (0 = prompt note)."""
    return (key + OFFSETS[pos % 4]) % 12


def _extract_pc(tok: torch.Tensor, tokenizer) -> int | None:
    """
    Extract pitch class from one preprocessed subbeat tensor (1, subseq_len).
    Tokens are in processed form: even slots = prog token, odd slots = pitch+dur token.
    Returns None if no valid note (EOS / pad position).
    """
    prog = int(tok[0, 0])
    if prog >= 128:                      # eos_token or pad_token
        return None
    pitch_dur = int(tok[0, 1])
    if pitch_dur < 128:                  # invalid (shouldn't happen for a real note)
        return None
    return (pitch_dur % 128) % 12       # pitch class


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gen_pcs(base: RoFormerSymbolicTransformer,
             key: int, n_gen: int,
             device: torch.device, temperature: float) -> list[int | None]:
    """Generate n_gen beats from a 1-beat prompt and return pitch classes."""
    prompt  = _prompt_from_key(key, n_prompt_bars=0, device=device,
                               base=base, n_prompt_beats=1)
    sampled = _sample(base, prompt, 1 + n_gen, temperature, device,
                      show_progress=False)
    return [_extract_pc(t, base.tokenizer) for t in sampled[1:]]   # skip prompt


@torch.no_grad()
def rule_following_acc(
    base: RoFormerSymbolicTransformer,
    keys: list[int],
    n_gen: int,
    n_trials: int,
    device: torch.device,
    temperature: float,
) -> tuple[dict[int, float], list[tuple[int, int | None]]]:
    """
    Returns
    -------
    per_key : {key: mean_accuracy}
    errors  : list of (expected_pc, got_pc) for every wrong position
    """
    per_key: dict[int, float] = {}
    errors: list[tuple[int, int | None]] = []

    for key in keys:
        trial_accs = []
        for _ in range(n_trials):
            pcs = _gen_pcs(base, key, n_gen, device, temperature)
            n_correct = 0
            for pos, pc in enumerate(pcs):
                exp = _expected_pc(key, pos + 1)   # pos+1 because pos=0 is the prompt
                if pc == exp:
                    n_correct += 1
                else:
                    errors.append((exp, pc))
            trial_accs.append(n_correct / len(pcs))
        per_key[key] = float(np.mean(trial_accs))

    return per_key, errors


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_summary_table(rows: list, n_trials: int) -> None:
    multi = n_trials > 1
    w = 10
    header = f"{'Key group':<20} {'Keys':<22} {'Accuracy':>{w}}"
    sep    = '-' * len(header)
    print()
    print('=' * len(header))
    print(f"RULE-FOLLOWING ACCURACY  (1-beat prompt {'greedy' if not multi else f't={n_trials} trials'})")
    print('=' * len(header))
    print(header)
    print(sep)
    all_means = []
    for cat, _, keys_str, mean, std in rows:
        all_means.append(mean)
        cell = f'{mean:.3f}±{std:.3f}' if multi else f'{mean:.3f}'
        print(f'{cat:<20} {keys_str:<22} {cell:>{w}}')
    print(sep)
    print(f"{'Overall':<20} {'all 12 keys':<22} {float(np.mean(all_means)):>{w}.3f}")
    print('=' * len(header))


def _print_per_key(rows: list) -> None:
    print()
    print('Per-key breakdown')
    print(f"  {'Key':>5}  {'Group':<16}  {'Acc':>5}")
    print('  ' + '-' * 32)
    for cat, per_key, *_ in rows:
        for key in sorted(per_key):
            print(f"  {ROOT_NAMES[key]:>5}  {cat:<16}  {per_key[key]:>5.3f}")


def _print_qualitative(base, keys, cat_label, n_gen, device, temperature,
                       show_beats=16) -> None:
    print(f'\n  [{cat_label}]')
    for key in keys:
        pcs  = _gen_pcs(base, key, n_gen, device, temperature)
        show = min(show_beats, n_gen)

        beat_row = '  '.join(f'{i+1:>4}' for i in range(show))
        exp_row  = '  '.join(f'{ROOT_NAMES[_expected_pc(key, i+1)]:>4}' for i in range(show))
        got_row  = '  '.join(f'{ROOT_NAMES[pc] if pc is not None else "?":>4}'
                             for pc in pcs[:show])
        mark_row = '  '.join(
            f'{"P" if i == -1 else ("✓" if pcs[i] == _expected_pc(key, i+1) else "✗"):>4}'
            for i in range(show)
        )
        acc = sum(1 for i, pc in enumerate(pcs) if pc == _expected_pc(key, i+1)) / len(pcs)

        print(f'    Key={ROOT_NAMES[key]:<3}  acc={acc:.3f}  '
              f'(prompt=[{ROOT_NAMES[key]}])')
        print(f'      Beat   : {beat_row}')
        print(f'      Expect : {exp_row}')
        print(f'      Got    : {got_row}')
        print(f'      Match  : {mark_row}')


def _print_error_dist(errors: list) -> None:
    n_total = len(errors)
    if n_total == 0:
        print('\nNo errors — perfect accuracy!')
        return

    pair_cnt: Counter = Counter()
    for exp, got in errors:
        pair_cnt[(exp, got if got is not None else -1)] += 1

    offset_cnt: Counter = Counter()
    for exp, got in errors:
        if got is not None:
            offset_cnt[(got - exp) % 12] += 1

    print(f'\nError distribution  ({n_total} wrong predictions)')
    print(f"  {'Expected':>8}  {'Got':>6}  {'Count':>6}  {'%':>6}")
    print('  ' + '-' * 32)
    for (exp, got), cnt in pair_cnt.most_common(20):
        got_name = ROOT_NAMES[got] if got >= 0 else 'none'
        print(f'  {ROOT_NAMES[exp]:>8}  {got_name:>6}  {cnt:>6}  {100*cnt/n_total:>5.1f}%')

    if offset_cnt:
        max_cnt = max(offset_cnt.values())
        print()
        print('  Pitch offset (predicted − expected) mod 12:')
        for off in range(12):
            cnt = offset_cnt.get(off, 0)
            if cnt == 0:
                continue
            bar = '█' * (cnt * 24 // max_cnt)
            print(f'    +{off:2d}  {cnt:5d}  {bar}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, model_size: int,
               device: torch.device) -> RoFormerSymbolicTransformer:
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    base   = RoFormerSymbolicTransformer(
        size=model_size, max_lr=max_lr, with_velocity=False
    )
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        base.load_state_dict(state)
        print(f'  Loaded : {ckpt_path}')
    else:
        print(f'  WARNING: {ckpt_path} not found — using random weights')
    return base.to(device).eval()


def run_evaluation(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('CP Bass Transformer — Rule-Following Evaluation')
    print('=' * 60)
    print(f'  checkpoint  : {args.base_ckpt}')
    print(f'  n_gen_beats : {args.n_gen_beats}  ({args.n_gen_beats // SUBBEATS_PER_BAR} bars)')
    print(f'  n_trials    : {args.n_trials}')
    print(f'  temperature : {args.temperature}')
    print(f'  prompt      : 1 note (starter x only — mirrors RASP prompt_len=1)')
    print()
    print(f'  Pretrain keys : {PRETRAIN_KEYS}')
    print(f'  Finetune-new  : {FINETUNE_NEW}')
    print(f'  Unseen        : {UNSEEN}')

    base = load_model(args.base_ckpt, args.model_size, device)

    rows       = []
    all_errors = []

    for cat, keys, keys_str in CATEGORIES:
        per_key, errors = rule_following_acc(
            base, keys, args.n_gen_beats, args.n_trials, device, args.temperature
        )
        all_errors.extend(errors)
        vals = list(per_key.values())
        rows.append((cat, per_key, keys_str,
                     float(np.mean(vals)), float(np.std(vals))))

    _print_summary_table(rows, args.n_trials)

    if args.verbose:
        _print_per_key(rows)

    # Qualitative examples (first 2 keys per category)
    print()
    print('Qualitative generation examples  (1-beat prompt, first 16 beats shown)')
    print('-' * 72)
    for cat, keys, _ in CATEGORIES:
        _print_qualitative(base, keys[:2], cat,
                           args.n_gen_beats, device, args.temperature)

    _print_error_dist(all_errors)


def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate CP bass transformer rule-following accuracy'
    )
    p.add_argument('--base_ckpt',   required=True,
                   help='.pt or .ckpt checkpoint path')
    p.add_argument('--model_size',  type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--n_gen_beats', type=int, default=32,
                   help='Beats to generate per trial (default 32 = 8 bars)')
    p.add_argument('--n_trials',    type=int, default=1,
                   help='Trials per key (useful with temperature > 0)')
    p.add_argument('--temperature', type=float, default=0,
                   help='0 = greedy (default), >0 = stochastic')
    p.add_argument('--verbose',     action='store_true',
                   help='Print per-key accuracy breakdown')
    return p.parse_args()


if __name__ == '__main__':
    run_evaluation(get_args())
