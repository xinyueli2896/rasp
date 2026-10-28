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
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS
from midi_adapter.infer_cp_bass import _prompt_from_key, decode_output

# Re-use shared helpers from evaluate_cp_bass
from midi_adapter.evaluate_cp_bass import (
    ROOT_NAMES,
    PRETRAIN_ONLY, FINETUNE_ONLY, BOTH_SEEN, UNSEEN, CATEGORIES,
    PRETRAIN_KEYS, FINETUNE_NEW, ALL_SEEN,   # backward-compat aliases
    _expected_pc, _extract_pc, _scored_positions,
    _print_summary_table, _print_per_key, _print_error_dist,
)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

# Prompts are seeded by (key, trial) so every model is scored on and writes
# MIDI from the SAME prompts. Without this each stem drew fresh realisations,
# which left the accuracy comparison unpaired and made side-by-side MIDI
# compare different prompts rather than different models.
PROMPT_SEED_BASE = 20261028


def prompt_seed(key: int, trial: int = 0) -> int:
    return PROMPT_SEED_BASE + 1000 * key + trial


@torch.no_grad()
def _gen_pcs(model: CPYinyangTransformer,
             key: int, n_gen: int,
             device: torch.device, temperature: float,
             n_prompt_beats: int = 2,
             max_polyphony: int = 4,
             beats_per_chord: int = 1,
             pattern: str | None = None,
             seed: int | None = None) -> list[int | None]:
    """Generate n_gen beats with rule conditioning and return pitch classes.

    pattern pins the prompt's realisation; None draws one at random per call.
    """
    prompt = _prompt_from_key(key, n_prompt_bars=0, device=device,
                              base=model.base, n_prompt_beats=n_prompt_beats,
                              max_polyphony=max_polyphony,
                              cycle_unit=('bar' if beats_per_chord > 1 else 'beat'),
                              pattern=pattern, seed=seed)

    total = n_prompt_beats + n_gen
    sampled = model.global_sampling(
        prompt, max_seq_len=total, temperature=temperature,
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
    beats_per_chord: int = 1,
    max_polyphony: int = 4,
) -> tuple[dict[int, float], list[tuple[int, int | None]]]:
    per_key: dict[int, float] = {}
    errors:  list[tuple[int, int | None]] = []

    for key in keys:
        trial_accs = []
        for _trial in range(n_trials):
            pcs = _gen_pcs(model, key, n_gen, device, temperature, n_prompt_beats,
                           max_polyphony, beats_per_chord,
                           seed=prompt_seed(key, _trial))
            total = n_prompt_beats + len(pcs)
            scored = _scored_positions(n_prompt_beats, total, beats_per_chord)
            n_correct = 0
            for t in scored:
                pc  = pcs[t - n_prompt_beats]
                exp = _expected_pc(key, t, beats_per_chord)
                if pc == exp:
                    n_correct += 1
                else:
                    errors.append((exp, pc))
            trial_accs.append(n_correct / max(len(scored), 1))
        per_key[key] = float(np.mean(trial_accs))

    return per_key, errors


# ---------------------------------------------------------------------------
# Qualitative output
# ---------------------------------------------------------------------------

def _print_qualitative(model, keys, cat_label, n_gen, device, temperature,
                       n_prompt_beats: int = 2, show_beats: int = 16,
                       beats_per_chord: int = 1) -> None:
    print(f'\n  [{cat_label}]')
    for key in keys:
        pcs  = _gen_pcs(model, key, n_gen, device, temperature, n_prompt_beats)
        show = min(show_beats, n_gen)

        prompt_notes = [ROOT_NAMES[_expected_pc(key, i, beats_per_chord)] for i in range(n_prompt_beats)]
        prompt_str   = ', '.join(prompt_notes)

        beat_row = '  '.join(f'{i+1:>4}' for i in range(show))
        exp_row  = '  '.join(
            f'{ROOT_NAMES[_expected_pc(key, i + n_prompt_beats, beats_per_chord)]:>4}' for i in range(show)
        )
        got_row  = '  '.join(
            f'{ROOT_NAMES[pc] if pc is not None else "?":>4}' for pc in pcs[:show]
        )
        mark_row = '  '.join(
            f'{"✓" if pcs[i] == _expected_pc(key, i + n_prompt_beats, beats_per_chord) else "✗":>4}'
            for i in range(show)
        )
        acc = sum(
            1 for i, pc in enumerate(pcs)
            if pc == _expected_pc(key, i + n_prompt_beats, beats_per_chord)
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
def _save_midi_all_keys(model, n_gen, device, temperature, n_prompt_beats, out_dir,
                        max_polyphony: int = 4, beats_per_chord: int = 1,
                        pattern: str | None = None, quiet: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_keys = list(range(12))
    for key in all_keys:
        prompt  = _prompt_from_key(key, n_prompt_bars=0, device=device,
                                   base=model.base, n_prompt_beats=n_prompt_beats,
                                   max_polyphony=max_polyphony,
                                   cycle_unit=('bar' if beats_per_chord > 1 else 'beat'),
                                   pattern=pattern,
                                   seed=prompt_seed(key))
        total   = n_prompt_beats + n_gen
        sampled = model.global_sampling(prompt, max_seq_len=total, temperature=temperature)
        path = os.path.join(out_dir, f'key_{ROOT_NAMES[key]}.mid')
        decode_output(sampled, model.base.tokenizer, save_path=path)
        if not quiet:
            print(f'  saved {path}')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _report_load(missing: list, unexpected: list) -> None:
    """Report BOTH sides of a strict=False load.

    Discarding `unexpected` hid the worst rebuild error this script can make:
    an adapter trained with n_skip=1 holds 12 yinyang_attn blocks, and
    rebuilding it at n_skip=4 builds 3. The 3 load, the other 9 land in
    `unexpected`, `missing` is empty -- and a wrong model scored a plausible
    table in silence. Adapter-block mismatches are called out by name.
    """
    import re as _re
    # Config buffers, not weights. A checkpoint written before they were made
    # persistent simply lacks them, and the value then comes from the command
    # line -- which the caller already prints. Warning about those is noise
    # that looks like a failure.
    CONFIG_BUFFERS = ('rule_subbeats_per_chord', 'rule_key_select_first',
                      'strict_causal')
    missing = [k for k in missing
               if not any(k.endswith(c) for c in CONFIG_BUFFERS)]
    if missing:
        print(f'  WARNING missing keys ({len(missing)}): {missing[:3]}')
    if unexpected:
        print(f'  WARNING unexpected keys ({len(unexpected)}): {unexpected[:3]}')
    def _n_adapters(keys):
        return len({int(m.group(1)) for k in keys
                    for m in [_re.search(r'yinyang_attn\.(\d+)\.', k)] if m})
    extra = _n_adapters(unexpected)
    if extra:
        print(f'  WARNING: the checkpoint carries {extra} adapter block(s) this '
              f'model does not have -- n_skip is almost certainly wrong, and '
              f'the loaded adapters are only a prefix of the trained ones.')


def load_model(base_ckpt: str | None, adapter_ckpt: str,
               model_size: int, adapter_rank: int, n_skip: int,
               bidirectional: bool, encoder_injected: bool,
               encoder_type: str = 'embedding', rule_mode: str = 'current',
               device: torch.device = None,
               rule_view: str = 'key_phase',
               lora_rank: int = 0,
               rule_input: str = 'tokens',
               rule_enc_type: str = 'mlp',
               rule_enc_hidden: int = 64,
               strict_causal: bool = False,
               rule_subbeats_per_chord: int = 1,
               rule_key_select: str = 'phase0') -> CPYinyangTransformer:
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    base  = RoFormerSymbolicTransformer(size=model_size, max_lr=max_lr, with_velocity=False)
    model = CPYinyangTransformer(base, adapter_rank=adapter_rank, n_skip=n_skip,
                                 bidirectional=bidirectional,
                                 encoder_injected=encoder_injected,
                                 encoder_type=encoder_type,
                                 rule_mode=rule_mode,
                                 rule_view=rule_view,
                                 lora_rank=lora_rank,
                                 rule_input=rule_input,
                                 rule_enc_type=rule_enc_type,
                                 rule_enc_hidden=rule_enc_hidden,
                                 strict_causal=strict_causal,
                                 rule_subbeats_per_chord=rule_subbeats_per_chord,
                                 rule_key_select=rule_key_select)

    if not (adapter_ckpt and os.path.exists(adapter_ckpt)):
        print(f'  WARNING: adapter ckpt not found ({adapter_ckpt}) — random weights')
        return model.to(device).eval()

    raw = torch.load(adapter_ckpt, map_location='cpu')

    if 'state_dict' in raw:
        # Lightning .ckpt — contains full model (base + adapter)
        full_state = {k[len('model.'):]: v for k, v in raw['state_dict'].items()
                      if k.startswith('model.')}
        missing, unexpected = model.load_state_dict(full_state, strict=False)
        _report_load(missing, unexpected)
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
        missing, unexpected = model.load_state_dict(raw, strict=False)
        _report_load(missing, unexpected)
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

    # Recover rule_mode/rule_view from the checkpoint's k_proj width rather than
    # trusting a flag: the view sets rule_d_model, and a mismatch rebuilds the
    # adapter at the wrong width so its weights silently fail to load. 16 is the
    # analytical model; an unknown width is a hard error, not a quiet rebuild.
    from models.rasp_compiled_rule import VIEW_WIDTH
    rule_mode, rule_view = args.rule_mode, (args.rule_view or 'key_phase')
    if args.adapter_ckpt and os.path.exists(args.adapter_ckpt):
        _raw = torch.load(args.adapter_ckpt, map_location='cpu')
        _sd  = _raw.get('state_dict', _raw) if isinstance(_raw, dict) else {}
        for k, v in _sd.items():
            if k.endswith('k_proj.weight'):
                w = v.shape[1]
                if w != 16:
                    hit = [n for n, ww in VIEW_WIDTH.items() if ww == w]
                    if not hit:
                        raise SystemExit(
                            f'{args.adapter_ckpt}: k_proj expects rule_hidden width '
                            f'{w}, neither 16 nor any known view {VIEW_WIDTH}')
                    rule_mode, rule_view = 'compiled', hit[0]
                    print(f'  inferred      : rule_mode=compiled '
                          f'rule_view={rule_view}  (k_proj width {w})')
                break

    model = load_model(getattr(args, 'base_ckpt', None), args.adapter_ckpt,
                       args.model_size, args.adapter_rank, args.n_skip,
                       args.bidirectional, args.encoder_injected,
                       args.encoder_type, rule_mode, device,
                       rule_view=rule_view)

    rows       = []
    all_errors = []

    for cat, keys, keys_str in CATEGORIES:
        per_key, errors = rule_following_acc(
            model, keys, args.n_gen_beats, args.n_trials,
            device, args.temperature, args.n_prompt_beats,
            beats_per_chord=args.beats_per_chord,
            max_polyphony=args.max_polyphony,
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

    midi_dir = args.save_midi_dir or os.path.join(
        'eval_midi', os.path.splitext(os.path.basename(args.adapter_ckpt))[0]
    )
    print(f'\nSaving MIDI for all 12 keys → {midi_dir}')
    _save_midi_all_keys(model, args.n_gen_beats, device, args.temperature,
                        args.n_prompt_beats, midi_dir,
                        max_polyphony=args.max_polyphony,
                        beats_per_chord=args.beats_per_chord)


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
    p.add_argument('--max_polyphony', type=int, default=4,
                   help='Note-slots per beat in the TRAINING data; must match '
                        'or the prompt is out of distribution.')
    p.add_argument('--beats_per_chord', type=int, default=1,
                   help='Beats the root is held for. 1 (default) scores every '
                        'beat. 4 matches --cycle_unit bar: only the DOWNBEAT '
                        'is pinned to the root, so only downbeats are scored.')
    p.add_argument('--temperature',   type=float, default=0)
    p.add_argument('--verbose',        action='store_true')
    p.add_argument('--bidirectional',    action='store_true',
                   help='Must match the flag used during training (no-input-to-rule-model variant)')
    p.add_argument('--encoder_injected', action='store_true',
                   help='Must match the flag used during training')
    p.add_argument('--encoder_type', type=str, default='embedding',
                   choices=['embedding', 'token_mlp'],
                   help='Must match the flag used during training')
    p.add_argument('--rule_view', type=str, default=None,
                   choices=['root_next', 'key_phase', 'computed', 'full'],
                   help='Only with --rule_mode compiled. Omit to infer it from the '
                        'checkpoint, which is safer than restating it.')
    p.add_argument('--rule_mode', type=str, default='current',
                   choices=['current', 'period4', 'seed_broadcast', 'compiled'],
                   help='Must match training. Normally unnecessary: a compiled '
                        'checkpoint is detected from its k_proj width.')
    p.add_argument('--save_midi_dir',  type=str, default=None,
                   help='Directory for MIDI output (default: eval_midi/<adapter_name>/)')
    return p.parse_args()


if __name__ == '__main__':
    run_evaluation(get_args())
