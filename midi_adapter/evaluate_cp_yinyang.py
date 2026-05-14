"""
Evaluate rule-following accuracy of the CPYinyangTransformer (base + adapter).

Mirrors evaluate_cp_bass.py but uses chord-conditioned autoregressive
generation via global_sampling_chord.

Usage
-----
  python -m midi_adapter.evaluate_cp_yinyang \\
      --base_ckpt    ckpt/cp_bass_ft_size1_batch8/last.ckpt \\
      --adapter_ckpt ckpt/cp_yinyang_size1_rank256/cp_yinyang_size1_rank256.pt

  # stochastic, save MIDI
  python -m midi_adapter.evaluate_cp_yinyang \\
      --base_ckpt    ckpt/... --adapter_ckpt ckpt/... \\
      --temperature 0.8 --n_trials 8 --save_midi_dir eval_midi_adapter/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.chord_tokenizer import chord_str_to_token
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS
from midi_adapter.infer_cp_bass import _prompt_from_key, decode_output

# Re-use shared helpers from evaluate_cp_bass
from midi_adapter.evaluate_cp_bass import (
    ROOT_NAMES,
    PRETRAIN_KEYS, FINETUNE_NEW, ALL_SEEN, UNSEEN, CATEGORIES,
    _expected_pc, _extract_pc,
    _print_summary_table, _print_per_key, _print_error_dist,
)


# ---------------------------------------------------------------------------
# Chord token helpers
# ---------------------------------------------------------------------------

def _make_chord_tokens(key: int, n_beats: int, device: torch.device) -> torch.Tensor:
    """Build beat-level chord token tensor for a given key and length."""
    tokens = [
        chord_str_to_token(f'{ROOT_NAMES[(key + OFFSETS[t % 4]) % 12]}:maj')
        for t in range(n_beats)
    ]
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, n_beats)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gen_pcs(model: CPYinyangTransformer,
             key: int, n_gen: int,
             device: torch.device, temperature: float,
             n_prompt_beats: int = 2) -> list[int | None]:
    """Generate n_gen beats with chord conditioning and return pitch classes."""
    prompt = _prompt_from_key(key, n_prompt_bars=0, device=device,
                              base=model.base, n_prompt_beats=n_prompt_beats)

    total     = n_prompt_beats + n_gen
    chord_tok = _make_chord_tokens(key, total, device)   # (1, total)

    sampled = model.global_sampling_chord(
        prompt, chord_tok, max_seq_len=total, temperature=temperature,
    )
    return [_extract_pc(t, model.base.tokenizer) for t in sampled[n_prompt_beats:]]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def rule_following_acc(
    model: CPYinyangTransformer,
    keys: list[int],
    n_gen: int,
    n_trials: int,
    device: torch.device,
    temperature: float,
    n_prompt_beats: int = 2,
) -> tuple[dict[int, float], list[tuple[int, int | None]]]:
    per_key: dict[int, float] = {}
    errors:  list[tuple[int, int | None]] = []

    for key in keys:
        trial_accs = []
        for _ in range(n_trials):
            pcs = _gen_pcs(model, key, n_gen, device, temperature, n_prompt_beats)
            n_correct = 0
            for pos, pc in enumerate(pcs):
                exp = _expected_pc(key, pos + n_prompt_beats)
                if pc == exp:
                    n_correct += 1
                else:
                    errors.append((exp, pc))
            trial_accs.append(n_correct / len(pcs))
        per_key[key] = float(np.mean(trial_accs))

    return per_key, errors


# ---------------------------------------------------------------------------
# Qualitative output
# ---------------------------------------------------------------------------

def _print_qualitative(model, keys, cat_label, n_gen, device, temperature,
                       n_prompt_beats: int = 2, show_beats: int = 16) -> None:
    print(f'\n  [{cat_label}]')
    for key in keys:
        pcs  = _gen_pcs(model, key, n_gen, device, temperature, n_prompt_beats)
        show = min(show_beats, n_gen)

        prompt_notes = [ROOT_NAMES[_expected_pc(key, i)] for i in range(n_prompt_beats)]
        prompt_str   = ', '.join(prompt_notes)

        beat_row = '  '.join(f'{i+1:>4}' for i in range(show))
        exp_row  = '  '.join(
            f'{ROOT_NAMES[_expected_pc(key, i + n_prompt_beats)]:>4}' for i in range(show)
        )
        got_row  = '  '.join(
            f'{ROOT_NAMES[pc] if pc is not None else "?":>4}' for pc in pcs[:show]
        )
        mark_row = '  '.join(
            f'{"✓" if pcs[i] == _expected_pc(key, i + n_prompt_beats) else "✗":>4}'
            for i in range(show)
        )
        acc = sum(
            1 for i, pc in enumerate(pcs)
            if pc == _expected_pc(key, i + n_prompt_beats)
        ) / len(pcs)

        print(f'    Key={ROOT_NAMES[key]:<3}  acc={acc:.3f}  (prompt=[{prompt_str}])')
        print(f'      Beat   : {beat_row}')
        print(f'      Expect : {exp_row}')
        print(f'      Got    : {got_row}')
        print(f'      Match  : {mark_row}')


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

@torch.no_grad()
def _save_midi_all_keys(model, n_gen, device, temperature, n_prompt_beats, out_dir) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_keys = sorted(set(PRETRAIN_KEYS + FINETUNE_NEW + UNSEEN))
    for key in all_keys:
        prompt    = _prompt_from_key(key, n_prompt_bars=0, device=device,
                                     base=model.base, n_prompt_beats=n_prompt_beats)
        total     = n_prompt_beats + n_gen
        chord_tok = _make_chord_tokens(key, total, device)
        sampled   = model.global_sampling_chord(
            prompt, chord_tok, max_seq_len=total, temperature=temperature,
        )
        path = os.path.join(out_dir, f'key_{ROOT_NAMES[key]}.mid')
        decode_output(sampled, model.base.tokenizer, save_path=path)
        print(f'  saved {path}')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_ckpt: str | None, adapter_ckpt: str,
               model_size: int, adapter_rank: int, n_skip: int,
               device: torch.device) -> CPYinyangTransformer:
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    base  = RoFormerSymbolicTransformer(size=model_size, max_lr=max_lr, with_velocity=False)
    model = CPYinyangTransformer(base, adapter_rank=adapter_rank, n_skip=n_skip)

    if not (adapter_ckpt and os.path.exists(adapter_ckpt)):
        print(f'  WARNING: adapter ckpt not found ({adapter_ckpt}) — random weights')
        return model.to(device).eval()

    raw = torch.load(adapter_ckpt, map_location='cpu')

    if 'state_dict' in raw:
        # Lightning .ckpt — contains full model (base + adapter)
        full_state = {k[len('model.'):]: v for k, v in raw['state_dict'].items()
                      if k.startswith('model.')}
        missing, _ = model.load_state_dict(full_state, strict=False)
        if missing:
            print(f'  WARNING missing keys: {missing[:3]}')
        print(f'  Loaded (Lightning ckpt, base+adapter): {adapter_ckpt}')
    else:
        # Lightweight .pt — adapter weights only; need base ckpt separately
        if base_ckpt and os.path.exists(base_ckpt):
            bstate = torch.load(base_ckpt, map_location='cpu')
            if 'state_dict' in bstate:
                bstate = bstate['state_dict']
            base.load_state_dict(bstate)
            print(f'  Base model   : {base_ckpt}')
        else:
            print(f'  WARNING: adapter is adapter-only .pt but no base_ckpt — random base weights')
        missing, _ = model.load_state_dict(raw, strict=False)
        if missing:
            print(f'  WARNING missing keys: {missing[:3]}')
        print(f'  Adapter      : {adapter_ckpt}')

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('CPYinyangTransformer — Rule-Following Evaluation')
    print('=' * 60)
    print(f'  adapter_ckpt  : {args.adapter_ckpt}')
    if args.base_ckpt:
        print(f'  base_ckpt     : {args.base_ckpt}  (used only for adapter-only .pt)')
    print(f'  n_gen_beats   : {args.n_gen_beats}  ({args.n_gen_beats // SUBBEATS_PER_BAR} bars)')
    print(f'  n_prompt_beats: {args.n_prompt_beats}')
    print(f'  n_trials      : {args.n_trials}')
    print(f'  temperature   : {args.temperature}')
    print(f'  adapter_rank  : {args.adapter_rank}   n_skip={args.n_skip}')
    print()
    print(f'  Pretrain keys : {PRETRAIN_KEYS}')
    print(f'  Finetune-new  : {FINETUNE_NEW}')
    print(f'  Unseen        : {UNSEEN}')

    model = load_model(getattr(args, 'base_ckpt', None), args.adapter_ckpt,
                       args.model_size, args.adapter_rank, args.n_skip, device)

    rows       = []
    all_errors = []

    for cat, keys, keys_str in CATEGORIES:
        per_key, errors = rule_following_acc(
            model, keys, args.n_gen_beats, args.n_trials,
            device, args.temperature, args.n_prompt_beats,
        )
        all_errors.extend(errors)
        vals = list(per_key.values())
        rows.append((cat, per_key, keys_str,
                     float(np.mean(vals)), float(np.std(vals))))

    _print_summary_table(rows, args.n_trials, n_prompt_beats=args.n_prompt_beats)

    if args.verbose:
        _print_per_key(rows)

    print()
    print(f'Qualitative generation examples  ({args.n_prompt_beats}-beat prompt, first 16 beats shown)')
    print('-' * 72)
    for cat, keys, _ in CATEGORIES:
        _print_qualitative(model, keys[:2], cat,
                           args.n_gen_beats, device, args.temperature,
                           n_prompt_beats=args.n_prompt_beats)

    _print_error_dist(all_errors)

    if args.save_midi_dir:
        print(f'\nSaving MIDI for all 12 keys → {args.save_midi_dir}')
        _save_midi_all_keys(model, args.n_gen_beats, device, args.temperature,
                            args.n_prompt_beats, args.save_midi_dir)


def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate CPYinyangTransformer rule-following accuracy'
    )
    p.add_argument('--base_ckpt',     default=None,
                   help='Base CP transformer ckpt (only needed when adapter_ckpt is an adapter-only .pt)')
    p.add_argument('--adapter_ckpt',  required=True,
                   help='Adapter .pt saved by train_cp_yinyang.py')
    p.add_argument('--model_size',    type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--adapter_rank',  type=int, default=256)
    p.add_argument('--n_skip',        type=int, default=4)
    p.add_argument('--n_gen_beats',   type=int, default=16,
                   help='Beats to generate per trial (default 16 = 4 bars; '
                        'keep n_prompt_beats + n_gen_beats <= TRAIN_LENGTH=24)')
    p.add_argument('--n_prompt_beats', type=int, default=2)
    p.add_argument('--n_trials',      type=int, default=1)
    p.add_argument('--temperature',   type=float, default=0)
    p.add_argument('--verbose',       action='store_true')
    p.add_argument('--save_midi_dir', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    run_evaluation(get_args())
