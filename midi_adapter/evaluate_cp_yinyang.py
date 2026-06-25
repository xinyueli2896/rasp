"""
Evaluate rule-following accuracy of the CPYinyangTransformer (base + adapter).

Mirrors evaluate_cp_bass.py but uses rule-conditioned autoregressive
generation via global_sampling (no chord tokens required).

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
from midi_adapter.generate_synthetic_bass import (
    SUBBEATS_PER_BAR, OFFSETS, generate_song, _preprocess_pm,
)
from midi_adapter.infer_cp_bass import _prompt_from_key, decode_output

# Re-use shared helpers from evaluate_cp_bass
from midi_adapter.evaluate_cp_bass import (
    ROOT_NAMES,
    PRETRAIN_ONLY, FINETUNE_ONLY, BOTH_SEEN, UNSEEN, CATEGORIES,
    PRETRAIN_KEYS, FINETUNE_NEW, ALL_SEEN,   # backward-compat aliases
    _expected_pc, _extract_pc,
    _print_summary_table, _print_per_key, _print_error_dist,
)

# Major triad intervals — must match CPChordRuleModel.CHORD_INTERVALS
_MAJOR_INTERVALS = (0, 4, 7)


# ---------------------------------------------------------------------------
# Polyphonic prompt helper  (chord approach needs a multi-voice seed)
# ---------------------------------------------------------------------------

def _poly_prompt_from_key(
    key: int, device: torch.device,
    base: RoFormerSymbolicTransformer,
    n_prompt_beats: int = 2,
) -> torch.Tensor:
    """Build a polyphonic 4-voice piano prompt in the given key."""
    n_bars = max(1, -(-n_prompt_beats // SUBBEATS_PER_BAR))
    pm, _ = generate_song(n_bars=n_bars, key=key, polyphonic=True, quality='maj')
    data, _ = _preprocess_pm(pm, n_prompt_beats)
    data = data.unsqueeze(0).to(device)
    pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
    return base.preprocess(data, pitch_shift)   # (1, n_prompt_beats, 8)


# ---------------------------------------------------------------------------
# Chord coverage helpers
# ---------------------------------------------------------------------------

def _extract_all_pcs(tok: torch.Tensor, tokenizer) -> set:
    """Extract pitch-class set from ALL voices of one subbeat tensor (1, subseq_len)."""
    S    = tok.shape[1]
    pcs  = set()
    pad  = tokenizer.pad_token
    eos  = tokenizer.eos_token
    for v in range(S // 2):
        prog     = int(tok[0, 2 * v])
        if prog == pad or prog == eos:
            break
        pitch_dur = int(tok[0, 2 * v + 1])
        if pitch_dur == pad or pitch_dur == eos or pitch_dur < 128:
            continue
        pcs.add((pitch_dur % 128) % 12)
    return pcs


def _expected_chord_pcs(key: int, pos: int) -> set:
    """Expected major-triad pitch-class set at absolute beat pos."""
    root = _expected_pc(key, pos)
    return {(root + i) % 12 for i in _MAJOR_INTERVALS}


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gen_beats(
    model: CPYinyangTransformer,
    key: int, n_gen: int,
    device: torch.device, temperature: float,
    n_prompt_beats: int = 2,
) -> list:
    """Generate n_gen beats; returns list of (1, subseq_len) tensors."""
    if model.approach == 'chord':
        prompt = _poly_prompt_from_key(key, device, model.base, n_prompt_beats)
    else:
        prompt = _prompt_from_key(key, n_prompt_bars=0, device=device,
                                  base=model.base, n_prompt_beats=n_prompt_beats)
    total   = n_prompt_beats + n_gen
    sampled = model.global_sampling(prompt, max_seq_len=total, temperature=temperature)
    return sampled[n_prompt_beats:]


@torch.no_grad()
def _gen_pcs(model: CPYinyangTransformer,
             key: int, n_gen: int,
             device: torch.device, temperature: float,
             n_prompt_beats: int = 2) -> list[int | None]:
    """Generate n_gen beats and return voice-0 pitch classes (bass note)."""
    beats = _gen_beats(model, key, n_gen, device, temperature, n_prompt_beats)
    return [_extract_pc(t, model.base.tokenizer) for t in beats]


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
    """Bass-note accuracy: voice-0 pitch class == expected root."""
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


@torch.no_grad()
def chord_coverage_acc(
    model: CPYinyangTransformer,
    keys: list[int],
    n_gen: int,
    n_trials: int,
    device: torch.device,
    temperature: float,
    n_prompt_beats: int = 2,
) -> dict[int, float]:
    """Chord coverage accuracy: fraction of beats where ALL expected major-triad
    pitch classes are present among ALL generated voices.

    Expected chord at beat t: {root, root+4, root+7} where root = (key+OFFSETS[t%4])%12.
    A beat is covered if expected_pcs ⊆ generated_pcs (extra notes are allowed).
    """
    per_key: dict[int, float] = {}

    for key in keys:
        trial_accs = []
        for _ in range(n_trials):
            beats = _gen_beats(model, key, n_gen, device, temperature, n_prompt_beats)
            n_covered = 0
            for pos, beat in enumerate(beats):
                gen_pcs  = _extract_all_pcs(beat, model.base.tokenizer)
                exp_pcs  = _expected_chord_pcs(key, pos + n_prompt_beats)
                if exp_pcs.issubset(gen_pcs):
                    n_covered += 1
            trial_accs.append(n_covered / len(beats))
        per_key[key] = float(np.mean(trial_accs))

    return per_key


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
    all_keys = list(range(12))
    for key in all_keys:
        if model.approach == 'chord':
            prompt = _poly_prompt_from_key(key, device, model.base, n_prompt_beats)
        else:
            prompt = _prompt_from_key(key, n_prompt_bars=0, device=device,
                                      base=model.base, n_prompt_beats=n_prompt_beats)
        total   = n_prompt_beats + n_gen
        sampled = model.global_sampling(prompt, max_seq_len=total, temperature=temperature)
        path = os.path.join(out_dir, f'key_{ROOT_NAMES[key]}.mid')
        decode_output(sampled, model.base.tokenizer, save_path=path)
        print(f'  saved {path}')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_ckpt: str | None, adapter_ckpt: str,
               model_size: int, adapter_rank: int, n_skip: int,
               bidirectional: bool, encoder_injected: bool,
               encoder_type: str = 'embedding', rule_mode: str = 'current',
               approach: str = 'bass',
               device: torch.device = None) -> CPYinyangTransformer:
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    base  = RoFormerSymbolicTransformer(size=model_size, max_lr=max_lr, with_velocity=False)
    model = CPYinyangTransformer(base, adapter_rank=adapter_rank, n_skip=n_skip,
                                 bidirectional=bidirectional,
                                 encoder_injected=encoder_injected,
                                 encoder_type=encoder_type,
                                 rule_mode=rule_mode,
                                 approach=approach)

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
    print(f'  bidirectional : {args.bidirectional}')
    print()
    print(f'  Pretrain keys : {PRETRAIN_KEYS}')
    print(f'  Finetune-new  : {FINETUNE_NEW}')
    print(f'  Unseen        : {UNSEEN}')

    model = load_model(getattr(args, 'base_ckpt', None), args.adapter_ckpt,
                       args.model_size, args.adapter_rank, args.n_skip,
                       args.bidirectional, args.encoder_injected,
                       args.encoder_type, args.rule_mode, args.approach, device)

    is_chord = (args.approach == 'chord')

    # --- Bass-note accuracy (voice 0 == expected root) ---
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

    print('\n── Bass-note accuracy (voice 0 == expected root) ──')
    _print_summary_table(rows, args.n_trials, n_prompt_beats=args.n_prompt_beats)
    if args.verbose:
        _print_per_key(rows)

    # --- Chord coverage accuracy (all triad tones present across all voices) ---
    if is_chord:
        print('\n── Chord coverage accuracy (all {root,root+4,root+7} present) ──')
        cov_rows = []
        for cat, keys, keys_str in CATEGORIES:
            per_key = chord_coverage_acc(
                model, keys, args.n_gen_beats, args.n_trials,
                device, args.temperature, args.n_prompt_beats,
            )
            vals = list(per_key.values())
            cov_rows.append((cat, per_key, keys_str,
                             float(np.mean(vals)), float(np.std(vals))))
        _print_summary_table(cov_rows, args.n_trials, n_prompt_beats=args.n_prompt_beats)
        if args.verbose:
            _print_per_key(cov_rows)

    print()
    print(f'Qualitative generation examples  ({args.n_prompt_beats}-beat prompt, first 16 beats shown)')
    print('-' * 72)
    for cat, keys, _ in CATEGORIES:
        _print_qualitative(model, keys[:2], cat,
                           args.n_gen_beats, device, args.temperature,
                           n_prompt_beats=args.n_prompt_beats)

    _print_error_dist(all_errors)

    midi_dir = args.save_midi_dir or os.path.join(
        'eval_midi', os.path.splitext(os.path.basename(args.adapter_ckpt))[0]
    )
    print(f'\nSaving MIDI for all 12 keys → {midi_dir}')
    _save_midi_all_keys(model, args.n_gen_beats, device, args.temperature,
                        args.n_prompt_beats, midi_dir)


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
    p.add_argument('--verbose',        action='store_true')
    p.add_argument('--bidirectional',    action='store_true',
                   help='Must match the flag used during training (no-input-to-rule-model variant)')
    p.add_argument('--encoder_injected', action='store_true',
                   help='Must match the flag used during training')
    p.add_argument('--encoder_type', type=str, default='embedding',
                   choices=['embedding', 'token_mlp'],
                   help='Must match the flag used during training')
    p.add_argument('--rule_mode', type=str, default='current',
                   choices=['current', 'period4', 'seed_broadcast'],
                   help='Must match the flag used during training')
    p.add_argument('--approach', type=str, default='bass',
                   choices=['bass', 'chord'],
                   help='Must match the flag used during training')
    p.add_argument('--save_midi_dir',  type=str, default=None,
                   help='Directory for MIDI output (default: eval_midi/<adapter_name>/)')
    return p.parse_args()


if __name__ == '__main__':
    run_evaluation(get_args())
