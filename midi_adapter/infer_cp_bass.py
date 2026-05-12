"""
Inference script for the pretrained CP bass transformer.

Generates bass MIDI by sampling autoregressively from the model.
Optionally conditions on a MIDI prompt file; otherwise starts from scratch.

Usage
-----
  # Generate 4 bars from scratch (key of C)
  python -m midi_adapter.infer_cp_bass \\
      --base_ckpt checkpoints/cp_bass_size1_pretrain.pt \\
      --key 0 --n_bars 8 --out out.mid

  # Continue a MIDI prompt
  python -m midi_adapter.infer_cp_bass \\
      --base_ckpt checkpoints/cp_bass_size1_pretrain.pt \\
      --prompt_mid prompt.mid --n_bars 8 --out out.mid

  # Different temperature / key
  python -m midi_adapter.infer_cp_bass \\
      --base_ckpt checkpoints/cp_bass_size1_pretrain.pt \\
      --key 7 --n_bars 16 --temperature 0.9 --out out.mid
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer, CPTokenizer
from midi_adapter.generate_synthetic_bass import (
    generate_song, _preprocess_pm,
    SUBBEATS_PER_BAR, CONSTANT_TEMPO, SECONDS_PER_SUBBEAT,
    DURATION_TEMPLATES, DURATION_BOUNDARIES,
)

try:
    import pretty_midi
except ImportError:
    print('pretty_midi is required: pip install pretty_midi')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Decode sampled token list → PrettyMIDI
# ---------------------------------------------------------------------------

def decode_output(
    sampled: list[torch.Tensor],   # list of (1, subseq_len) tensors, one per subbeat
    tokenizer: CPTokenizer,
    tempo: float = CONSTANT_TEMPO,
    velocity: int = 100,
    fixed_program: int | None = None,
    save_path: str | None = None,
) -> pretty_midi.PrettyMIDI:
    """
    Decode sampled token list to PrettyMIDI (matches original repo's decode_output).

    Token pairs per subbeat: (program, pitch_enc) where
      pitch_enc = pitch + (duration+1)*128, so pitch_enc - 128 = pitch + duration*128
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = SECONDS_PER_SUBBEAT

    instrument_map: dict[int, pretty_midi.Instrument] = {}

    for time_step, tok_batch in enumerate(sampled):
        content = tok_batch[0]                     # (subseq_len,)
        start_time = time_step * time_step_length

        for i in range(0, len(content), 2):
            program = int(content[i].item())
            if program == tokenizer.eos_token:
                break
            if i + 1 >= len(content):
                break

            pitch_duration = int(content[i + 1].item()) - 128
            pitch    = pitch_duration % 128
            duration = pitch_duration // 128

            if program < 0 or program >= 128:
                break
            if not (0 <= pitch < 128):
                break
            if not (0 <= duration < len(DURATION_TEMPLATES)):
                break

            end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time

            if program not in instrument_map:
                if program == 127:
                    instrument_map[program] = pretty_midi.Instrument(0, is_drum=True)
                else:
                    prog = fixed_program if fixed_program is not None else program
                    instrument_map[program] = pretty_midi.Instrument(program=prog)
                pm.instruments.append(instrument_map[program])

            instrument_map[program].notes.append(
                pretty_midi.Note(velocity=velocity, pitch=pitch,
                                 start=start_time, end=end_time)
            )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        pm.write(save_path)

    return pm


# ---------------------------------------------------------------------------
# Build prompt tensor
# ---------------------------------------------------------------------------

def _prompt_from_midi(path: str, n_prompt_bars: int, device: torch.device,
                      base: RoFormerSymbolicTransformer) -> torch.Tensor:
    """Load a MIDI file and preprocess it into a prompt tensor."""
    pm = pretty_midi.PrettyMIDI(path)
    n_subbeats = n_prompt_bars * SUBBEATS_PER_BAR
    data, _ = _preprocess_pm(pm, n_subbeats)
    data = data.unsqueeze(0).to(device)                  # (1, T, 16)
    pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
    return base.preprocess(data, pitch_shift)            # (1, T, 8)


def _prompt_from_key(key: int, n_prompt_bars: int, device: torch.device,
                     base: RoFormerSymbolicTransformer,
                     n_prompt_beats: int | None = None) -> torch.Tensor:
    """Generate a synthetic prompt in the given key.
    n_prompt_beats overrides n_prompt_bars if given (finer granularity)."""
    n_subbeats = (n_prompt_beats if n_prompt_beats is not None
                  else n_prompt_bars * SUBBEATS_PER_BAR)
    if n_subbeats == 0:
        dummy = torch.zeros(1, 1, 16, dtype=torch.uint8, device=device)
        pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
        return base.preprocess(dummy, pitch_shift)

    pm, _ = generate_song(n_bars=max(1, -(-n_subbeats // SUBBEATS_PER_BAR)), key=key)
    data, _ = _preprocess_pm(pm, n_subbeats)
    data = data.unsqueeze(0).to(device)
    pitch_shift = torch.zeros(1, dtype=torch.long, device=device)
    return base.preprocess(data, pitch_shift)


# ---------------------------------------------------------------------------
# Local sampling without KV cache (RoFormerEncoder doesn't expose it)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _local_sample(
    base: RoFormerSymbolicTransformer,
    h: torch.Tensor,            # (B, H) global hidden for this subbeat
    temperature: float,
    max_len: int = 32,
) -> torch.Tensor:
    """Sample one subbeat's token sequence autoregressively (no KV cache)."""
    B = h.shape[0]
    tokenizer = base.tokenizer
    h_emb = h.unsqueeze(1)                                   # (B, 1, H) — first token
    generated: list[torch.Tensor] = []
    eos_triggered = torch.zeros(B, dtype=torch.bool, device=h.device)

    for _ in range(max_len):
        seq = torch.cat([h_emb] + [base.local_embedding(t.unsqueeze(1))
                                   for t in generated], dim=1)  # (B, t+1, H)
        attn_mask = base.buffered_future_mask(seq)
        out = base.local_decoder(seq, attention_mask=attn_mask)[0]  # (B, t+1, H)
        logits = base.final_decoder(out[:, -1])                     # (B, V)

        if temperature == 0:
            next_tok = logits.argmax(dim=-1, keepdim=True)          # (B, 1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)                  # (B, 1)

        next_tok[eos_triggered] = tokenizer.pad_token
        eos_triggered = eos_triggered | (next_tok.squeeze(1) == tokenizer.eos_token)
        generated.append(next_tok.squeeze(1))                       # (B,)

        if eos_triggered.all():
            break

    if not generated:
        return torch.full((B, 1), tokenizer.pad_token, dtype=torch.long, device=h.device)
    return torch.stack(generated, dim=1)                            # (B, seq_len)


# ---------------------------------------------------------------------------
# AR sampling loop (no KV cache — RoFormerEncoder doesn't expose it)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample(
    base: RoFormerSymbolicTransformer,
    prompt: torch.Tensor,
    total_subbeats: int,
    temperature: float,
    device: torch.device,
    show_progress: bool = True,
) -> list[torch.Tensor]:
    """
    Returns a list of (1, subseq_len) token tensors, one per subbeat,
    covering the full total_subbeats (prompt + generated).
    """
    B = 1
    sos = base.global_sos.view(1, 1, -1)                     # (1, 1, H)

    # Encode prompt
    prompt_len = prompt.shape[1]
    h_prompt, _ = base.local_encode(prompt)                  # (B*T, H)
    h_prompt = h_prompt.view(B, prompt_len, -1)

    # history: encoded hiddens, one per subbeat already processed
    h_hist = torch.cat([sos, h_prompt], dim=1)               # (1, 1+T, H)

    # Keep raw token arrays for the prompt subbeats (we'll decode everything)
    prompt_tokens = [prompt[:, t, :] for t in range(prompt_len)]
    gen_tokens: list[torch.Tensor] = []

    for step in range(prompt_len, total_subbeats):
        if show_progress and step % SUBBEATS_PER_BAR == 0:
            print(f'  subbeat {step}/{total_subbeats}')

        # Global transformer over everything seen so far
        attn_mask = base.buffered_future_mask(h_hist)
        h_out = base.model(h_hist, attention_mask=attn_mask)[0]  # (1, 1+step, H)
        h_last = h_out[:, -1, :]                                  # (1, H)

        # Local sampling for this subbeat (no KV cache)
        y_next = _local_sample(base, h_last, temperature)              # (1, subseq_len)
        gen_tokens.append(y_next)

        # Encode y_next and append to history
        h_enc, _ = base.local_encode(y_next.unsqueeze(1))       # (1, H)
        h_hist = torch.cat([h_hist, h_enc.unsqueeze(1)], dim=1)

    return prompt_tokens + gen_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    max_lr = 5e-5 if args.model_size >= 2 else 1e-4
    base   = RoFormerSymbolicTransformer(
        size=args.model_size, max_lr=max_lr, with_velocity=False,
    )

    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        base.load_state_dict(state)
        print(f'Loaded: {args.base_ckpt}')
    else:
        print('WARNING: no checkpoint found — using random weights.')

    base = base.to(device).eval()

    # Build prompt
    if args.prompt_mid and os.path.exists(args.prompt_mid):
        print(f'Using MIDI prompt: {args.prompt_mid} ({args.n_prompt_bars} bars)')
        prompt = _prompt_from_midi(args.prompt_mid, args.n_prompt_bars, device, base)
    else:
        key_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][args.key % 12]
        print(f'Generating synthetic prompt: key={key_name} ({args.n_prompt_bars} prompt bars)')
        prompt = _prompt_from_key(args.key, args.n_prompt_bars, device, base,
                                  n_prompt_beats=args.n_prompt_beats)

    prompt_len  = prompt.shape[1]
    total_subbeats = prompt_len + args.n_bars * SUBBEATS_PER_BAR
    print(f'Prompt subbeats: {prompt_len}  |  Total to generate: {total_subbeats}')

    # Sample — full-context AR loop (RoFormerEncoder doesn't expose KV cache)
    print(f'Sampling (temperature={args.temperature}) ...')
    sampled = _sample(base, prompt, total_subbeats, args.temperature, device)

    # Keep only the newly generated subbeats (after the prompt)
    generated = sampled[prompt_len:]

    # Decode to MIDI
    decode_output(generated, base.tokenizer, save_path=args.out,
                  fixed_program=args.fixed_program, velocity=args.velocity)
    print(f'Saved → {args.out}')


def get_args():
    p = argparse.ArgumentParser(description='CP bass transformer inference')
    p.add_argument('--base_ckpt',     type=str, required=True,
                   help='Path to pretrained CP transformer .pt or .ckpt file')
    p.add_argument('--out',           type=str, default='output.mid',
                   help='Output MIDI path')
    p.add_argument('--n_bars',        type=int, default=8,
                   help='Number of bars to generate (after prompt)')
    p.add_argument('--n_prompt_bars', type=int, default=1,
                   help='Bars of prompt to condition on (0 = unconditioned)')
    p.add_argument('--n_prompt_beats', type=int, default=None,
                   help='Beats (notes) of prompt — overrides --n_prompt_bars. Use 1 for single-note prompt.')
    p.add_argument('--key',           type=int, default=0,
                   help='Key root 0-11 for synthetic prompt (ignored if --prompt_mid given)')
    p.add_argument('--prompt_mid',    type=str, default=None,
                   help='MIDI file to use as prompt instead of synthetic')
    p.add_argument('--temperature',   type=float, default=1.0,
                   help='Sampling temperature (0 = greedy)')
    p.add_argument('--model_size',    type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--velocity',      type=int, default=100)
    p.add_argument('--fixed_program', type=int, default=None,
                   help='Force all notes to this MIDI program (e.g. 32 for bass)')
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    infer(args)
