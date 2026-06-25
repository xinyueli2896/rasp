"""
Generate music from a custom MIDI prompt using CPYinyangTransformer.

The adapter steers the pretrained CP transformer toward the I-IV-V-I cadence rule.
The key is automatically read from the first beat of your prompt (voice 0 lowest pitch).

Usage
-----
  # From your own MIDI file (chord approach, n_skip=1)
  python -m midi_adapter.infer_cp_yinyang \\
      --adapter_ckpt checkpoints/.../last.ckpt \\
      --prompt_midi  my_prompt.mid \\
      --n_prompt_beats 4 \\
      --approach chord --encoder_injected --n_skip 1 \\
      --n_gen_beats 32 --out output.mid

  # From a synthetic prompt (given key)
  python -m midi_adapter.infer_cp_yinyang \\
      --adapter_ckpt checkpoints/.../last.ckpt \\
      --key 0 --n_prompt_beats 4 \\
      --approach chord --encoder_injected --n_skip 1 \\
      --n_gen_beats 32 --out output.mid
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.generate_synthetic_bass import (
    SUBBEATS_PER_BAR, generate_song, _preprocess_pm, pitch_sort_cp,
)
from midi_adapter.infer_cp_bass import decode_output

try:
    import pretty_midi
except ImportError:
    print('pretty_midi is required: pip install pretty_midi')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _prompt_from_midi_file(
    path: str,
    n_prompt_beats: int,
    device: torch.device,
    base: RoFormerSymbolicTransformer,
    pitch_sort: bool = True,
) -> torch.Tensor:
    """Load a MIDI file and convert to a preprocessed CP prompt tensor.

    The key is automatically extracted from beat 0 voice 0 (lowest pitch).
    pitch_sort=True ensures voice 0 holds the lowest pitch at every beat,
    which is required for the chord approach to read the correct key.
    """
    pm = pretty_midi.PrettyMIDI(path)
    data, _ = _preprocess_pm(pm, n_prompt_beats)   # (n_prompt_beats, 16) uint8
    if pitch_sort:
        data = pitch_sort_cp(data)
    data        = data.unsqueeze(0).to(device)     # (1, n_prompt_beats, 16)
    pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
    prompt      = base.preprocess(data, pitch_shift)   # (1, n_prompt_beats, 8)
    return prompt


def _prompt_from_key_poly(
    key: int,
    n_prompt_beats: int,
    device: torch.device,
    base: RoFormerSymbolicTransformer,
) -> torch.Tensor:
    """Synthetic 4-voice piano prompt in the given key (chord approach)."""
    n_bars = max(1, -(-n_prompt_beats // SUBBEATS_PER_BAR))
    pm, _  = generate_song(n_bars=n_bars, key=key, polyphonic=True, quality='maj')
    data, _ = _preprocess_pm(pm, n_prompt_beats)
    data    = pitch_sort_cp(data)
    data    = data.unsqueeze(0).to(device)
    pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
    return base.preprocess(data, pitch_shift)


def _prompt_from_key_mono(
    key: int,
    n_prompt_beats: int,
    device: torch.device,
    base: RoFormerSymbolicTransformer,
) -> torch.Tensor:
    """Synthetic monophonic bass prompt in the given key (bass approach)."""
    n_bars = max(1, -(-n_prompt_beats // SUBBEATS_PER_BAR))
    pm, _  = generate_song(n_bars=n_bars, key=key, polyphonic=False)
    data, _ = _preprocess_pm(pm, n_prompt_beats)
    data    = data.unsqueeze(0).to(device)
    pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
    return base.preprocess(data, pitch_shift)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, device: torch.device) -> CPYinyangTransformer:
    max_lr = 5e-5 if args.model_size >= 2 else 1e-4
    base   = RoFormerSymbolicTransformer(
        size=args.model_size, max_lr=max_lr, with_velocity=False,
    )
    model  = CPYinyangTransformer(
        base,
        adapter_rank     = args.adapter_rank,
        n_skip           = args.n_skip,
        encoder_injected = args.encoder_injected,
        encoder_type     = args.encoder_type,
        rule_mode        = args.rule_mode,
        approach         = args.approach,
    )

    raw = torch.load(args.adapter_ckpt, map_location='cpu')
    if 'state_dict' in raw:
        full_state = {k[len('model.'):]: v for k, v in raw['state_dict'].items()
                      if k.startswith('model.')}
        missing, _ = model.load_state_dict(full_state, strict=False)
    else:
        if args.base_ckpt and os.path.exists(args.base_ckpt):
            bstate = torch.load(args.base_ckpt, map_location='cpu')
            if 'state_dict' in bstate:
                bstate = bstate['state_dict']
            base.load_state_dict(bstate)
            print(f'  Base model   : {args.base_ckpt}')
        missing, _ = model.load_state_dict(raw, strict=False)
    if missing:
        print(f'  WARNING missing keys: {missing[:3]}')
    print(f'  Adapter      : {args.adapter_ckpt}')
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  approach={args.approach}  |  n_skip={args.n_skip}')

    model = load_model(args, device)

    # Build prompt
    if args.prompt_midi and os.path.exists(args.prompt_midi):
        print(f'Prompt: {args.prompt_midi}  ({args.n_prompt_beats} beats)')
        prompt = _prompt_from_midi_file(
            args.prompt_midi, args.n_prompt_beats, device, model.base,
            pitch_sort=True,
        )
    else:
        key_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][args.key % 12]
        print(f'Synthetic prompt: key={key_name}  ({args.n_prompt_beats} beats)')
        if args.approach == 'chord':
            prompt = _prompt_from_key_poly(args.key, args.n_prompt_beats, device, model.base)
        else:
            prompt = _prompt_from_key_mono(args.key, args.n_prompt_beats, device, model.base)

    # Log inferred key
    inferred_key = int((prompt[:, 0, 1] % 128 % 12).clamp(0, 11).item())
    key_names    = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    print(f'Inferred key from prompt beat 0: {key_names[inferred_key]}  ({inferred_key})')

    n_prompt = prompt.shape[1]
    total    = n_prompt + args.n_gen_beats
    print(f'Generating {args.n_gen_beats} beats ({args.n_gen_beats // SUBBEATS_PER_BAR} bars) ...')

    sampled = model.global_sampling(
        prompt, max_seq_len=total, temperature=args.temperature,
    )

    # sampled[0..n_prompt-1] = prompt; sampled[n_prompt..] = generated
    generated = sampled if args.include_prompt else sampled[n_prompt:]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    decode_output(generated, model.base.tokenizer, save_path=args.out,
                  velocity=args.velocity)
    print(f'Saved → {args.out}')


def get_args():
    p = argparse.ArgumentParser(description='CPYinyangTransformer inference')
    p.add_argument('--adapter_ckpt',  type=str, required=True,
                   help='Adapter .ckpt or .pt saved by train_cp_yinyang.py')
    p.add_argument('--base_ckpt',     type=str, default=None,
                   help='Base CP transformer .pt (only needed for adapter-only .pt files)')
    p.add_argument('--prompt_midi',   type=str, default=None,
                   help='Your own MIDI file to use as prompt. Key is auto-detected '
                        'from beat 0 voice 0 (lowest pitch). If omitted, a synthetic '
                        'prompt is built from --key.')
    p.add_argument('--key',           type=int, default=0,
                   help='Key root 0-11 for synthetic prompt (ignored if --prompt_midi given)')
    p.add_argument('--n_prompt_beats', type=int, default=4,
                   help='Number of beats to use as prompt (default: 4 = 1 bar)')
    p.add_argument('--n_gen_beats',   type=int, default=32,
                   help='Number of beats to generate after the prompt (default: 32 = 8 bars)')
    p.add_argument('--out',           type=str, default='output.mid',
                   help='Output MIDI path')
    p.add_argument('--include_prompt', action='store_true',
                   help='Include prompt beats at the start of the output MIDI')
    p.add_argument('--temperature',   type=float, default=0.0,
                   help='Sampling temperature (0 = greedy, try 0.8-1.0 for variety)')
    p.add_argument('--velocity',      type=int, default=100)
    # Model architecture — must match training
    p.add_argument('--approach',      type=str, default='chord',
                   choices=['bass', 'chord'])
    p.add_argument('--encoder_injected', action='store_true')
    p.add_argument('--encoder_type', type=str, default='embedding',
                   choices=['embedding', 'token_mlp'])
    p.add_argument('--rule_mode',    type=str, default='current',
                   choices=['current', 'period4', 'seed_broadcast'])
    p.add_argument('--model_size',   type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip',       type=int, default=1)
    p.add_argument('--seed',         type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    infer(args)
