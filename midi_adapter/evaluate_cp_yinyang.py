"""
Evaluate chord root accuracy on four key groups:

  pretrain_keys  {0,2,4,5,7,9,11}       seen in CP pretraining only
  finetune_new   {1,3,10}               first seen in adapter finetuning
  all_seen       {0,1,2,3,4,5,7,9,10,11} seen in any training phase
  unseen         {6,8}                   never seen in training

Accuracy = fraction of bar-start positions (every 16th subbeat) where
  argmax(pitch-slot logits) % 128  ==  36 + chord_root

Can evaluate:
  - Base CP transformer alone (no adapter)
  - CPYinyangTransformer with adapter weights

Usage
-----
  # base model only
  python -m midi_adapter.evaluate_cp_yinyang \\
      --base_ckpt checkpoints/cp_bass_size1_pretrain.pt

  # with adapter
  python -m midi_adapter.evaluate_cp_yinyang \\
      --base_ckpt    checkpoints/cp_bass_size1_pretrain.pt \\
      --adapter_ckpt checkpoints/cp_yinyang_size1_rank256.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.chord_tokenizer import N_QUALITIES, NO_CHORD_TOKEN
from midi_adapter.generate_synthetic_bass import (
    generate_song, _preprocess_pm, SUBBEATS_PER_BAR, OFFSETS,
)

# ---------------------------------------------------------------------------
# Key groups
# ---------------------------------------------------------------------------

KEY_GROUPS = {
    'pretrain_keys': [0, 2, 4, 5, 7, 9, 11],
    'finetune_new':  [1, 3, 10],
    'all_seen':      [0, 1, 2, 3, 4, 5, 7, 9, 10, 11],
    'unseen':        [6, 8],
}

TRAIN_LENGTH     = 384   # subbeats per window
N_BARS_PER_WIN   = TRAIN_LENGTH // SUBBEATS_PER_BAR   # 24


# ---------------------------------------------------------------------------
# Generate a batch of test songs for a given key list
# ---------------------------------------------------------------------------

def make_batch(keys: list[int], n_songs: int, n_bars: int, device: torch.device):
    """
    Generate n_songs songs (one per key, cycling through keys) and return:
      x            (n_songs, n_bars*16, 16)  uint8 CP data
      chord_tokens (n_songs, n_bars)         int64 bar-level chord tokens
      pitch_shift  (n_songs,)                zeros (no augmentation at eval)
    """
    from midi_adapter.chord_tokenizer import chord_str_to_token
    from midi_adapter.generate_synthetic_bass import ROOT_NAMES, COMMON_CHORDS

    n_subbeats = n_bars * SUBBEATS_PER_BAR
    x_list, ct_list = [], []

    for i in range(n_songs):
        key = keys[i % len(keys)]
        pm, xf_chords = generate_song(n_bars=n_bars, key=key)
        data, _ = _preprocess_pm(pm, n_subbeats)
        x_list.append(data)

        # Build chord tokens for each bar
        bar_tokens = []
        for b in range(n_bars):
            root = (key + OFFSETS[b % 4]) % 12
            # quality is random per song in generate_song; extract from xf_chords
            chord_str = xf_chords[b][1]
            tok = chord_str_to_token(chord_str)
            bar_tokens.append(tok)
        ct_list.append(torch.tensor(bar_tokens, dtype=torch.long))

    x            = torch.stack(x_list, dim=0).to(device)          # (B, T, 16)
    chord_tokens = torch.stack(ct_list, dim=0).to(device)         # (B, n_bars)
    pitch_shift  = torch.zeros(n_songs, dtype=torch.long, device=device)
    return x, chord_tokens, pitch_shift


# ---------------------------------------------------------------------------
# Accuracy computation (teacher-forced)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(model, x, chord_tokens, pitch_shift, base, window_size=TRAIN_LENGTH):
    """
    Slide a window of `window_size` subbeats over each song (step = SUBBEATS_PER_BAR)
    and accumulate chord-root accuracy.

    For each bar-start subbeat within the window (positions 0, 16, 32, ...):
      pred_pitch   = argmax(logits[:, ::16, slot=1, :]) % 128
      target_pitch = 36 + chord_tokens // N_QUALITIES
    """
    B, T, _ = x.shape
    n_song_bars = T // SUBBEATS_PER_BAR
    n_win_bars  = window_size // SUBBEATS_PER_BAR

    correct = 0
    total   = 0

    for start_sub in range(0, T - window_size + 1, SUBBEATS_PER_BAR):
        start_bar = start_sub // SUBBEATS_PER_BAR
        end_bar   = start_bar + n_win_bars

        x_win  = x[:, start_sub:start_sub + window_size, :]        # (B, W, 16)
        ct_win = chord_tokens[:, start_bar:end_bar]                 # (B, n_win_bars)
        ps     = pitch_shift

        x_proc = base.preprocess(x_win, ps)                        # (B, W, 8)

        if isinstance(model, CPYinyangTransformer):
            logits = model(x_proc, ct_win)                         # (B*W, 8, V)
        else:
            # base CP transformer — no chord conditioning
            logits = base(x_proc)                                   # (B*W, 8, V)

        logits = logits.view(B, window_size, 8, -1)                # (B, W, 8, V)

        # Pitch predictions at bar-start subbeats
        pitch_logits   = logits[:, ::16, 1, :]                     # (B, n_win_bars, V)
        pred_enc       = pitch_logits.argmax(-1)                   # (B, n_win_bars)
        pred_pitch     = pred_enc % 128

        expected_root  = ct_win // N_QUALITIES                     # (B, n_win_bars)
        expected_pitch = 36 + expected_root

        valid   = ct_win != NO_CHORD_TOKEN
        correct += (pred_pitch == expected_pitch)[valid].sum().item()
        total   += valid.sum().item()

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    max_lr = 5e-5 if args.model_size >= 2 else 1e-4
    base   = RoFormerSymbolicTransformer(
        size=args.model_size, max_lr=max_lr, with_velocity=False,
    )

    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location='cpu')
        base.load_state_dict(state)
        print(f'Loaded base CP transformer: {args.base_ckpt}')
    else:
        print('WARNING: no base checkpoint — using random weights.')

    use_adapter = args.adapter_ckpt and os.path.exists(args.adapter_ckpt)
    if use_adapter:
        model = CPYinyangTransformer(base, adapter_rank=args.adapter_rank, n_skip=args.n_skip)
        state = torch.load(args.adapter_ckpt, map_location='cpu')
        model.load_state_dict(state)
        print(f'Loaded adapter: {args.adapter_ckpt}')
    else:
        model = base
        print('Evaluating base CP transformer (no adapter).')

    model = model.to(device).eval()
    base  = base.to(device).eval()

    # Header
    col_w = 18
    print()
    print(f'{"Key group":<{col_w}}  {"Keys":<30}  {"n_songs":>8}  {"accuracy":>10}')
    print('-' * 72)

    results = {}
    for group_name, keys in KEY_GROUPS.items():
        x, chord_tokens, pitch_shift = make_batch(
            keys, n_songs=args.n_songs, n_bars=args.n_bars, device=device,
        )
        acc = compute_accuracy(model, x, chord_tokens, pitch_shift, base,
                               window_size=TRAIN_LENGTH)
        results[group_name] = acc
        print(f'{group_name:<{col_w}}  {str(keys):<30}  {args.n_songs:>8}  {acc:>10.4f}')

    print()
    print('unseen / all_seen ratio: '
          f'{results["unseen"] / max(results["all_seen"], 1e-6):.4f}')

    return results


def get_args():
    p = argparse.ArgumentParser(description='Evaluate CPYinyangTransformer chord accuracy')
    p.add_argument('--base_ckpt',    type=str, required=True,
                   help='Path to pretrained CP transformer .pt file')
    p.add_argument('--adapter_ckpt', type=str, default=None,
                   help='Path to adapter .pt file (omit to eval base model only)')
    p.add_argument('--model_size',   type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip',       type=int, default=4)
    p.add_argument('--n_songs',      type=int, default=50,
                   help='Songs per key group')
    p.add_argument('--n_bars',       type=int, default=32,
                   help='Bars per song (>= 24 for at least one full window)')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(get_args())
