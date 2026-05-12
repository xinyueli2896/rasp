"""
Decode samples from a preprocessed CP dataset (.pt) back to MIDI.
Uses the same FramedDataset as training so you hear exactly what the model sees.

Usage
-----
  python -m midi_adapter.decode_dataset_to_midi \
      --data data/bass_pretrain_cp4.pt \
      --out_dir /tmp/decoded_midi \
      --n_samples 8 \
      --split train
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pretty_midi
except ImportError:
    print('pretty_midi required: pip install pretty_midi')
    sys.exit(1)

from cp_transformer import FramedDataset

# Must match generate_synthetic_bass.py
CONSTANT_TEMPO      = 120.0
BEAT_DIV            = 4
SECONDS_PER_SUBBEAT = 60.0 / CONSTANT_TEMPO / BEAT_DIV  # 0.125

DURATION_TEMPLATES = np.array([
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
    192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
])

MAX_POLYPHONY = 4


def decode_frame_to_midi(frame: torch.Tensor) -> pretty_midi.PrettyMIDI:
    """
    frame : (n_subbeats, MAX_POLYPHONY * 4) uint8 tensor
    Returns a PrettyMIDI with one instrument track per program seen.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=CONSTANT_TEMPO)
    pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]

    instruments: dict[int, pretty_midi.Instrument] = {}
    n_subbeats = frame.shape[0]

    for s in range(n_subbeats):
        t_start = s * SECONDS_PER_SUBBEAT
        for p in range(MAX_POLYPHONY):
            prog     = int(frame[s, p * 4 + 0])
            pitch    = int(frame[s, p * 4 + 1])
            dur_idx  = int(frame[s, p * 4 + 2])
            velocity = int(frame[s, p * 4 + 3])

            if prog == 255:   # empty slot
                break
            if prog == 254:   # EOS
                break

            dur_subbeats = int(DURATION_TEMPLATES[min(dur_idx, len(DURATION_TEMPLATES) - 1)])
            t_end = (s + dur_subbeats) * SECONDS_PER_SUBBEAT

            if prog not in instruments:
                instruments[prog] = pretty_midi.Instrument(program=prog)
            instruments[prog].notes.append(
                pretty_midi.Note(velocity=velocity, pitch=pitch,
                                 start=t_start, end=t_end)
            )

    for inst in instruments.values():
        pm.instruments.append(inst)

    return pm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       required=True, help='Path to .pt dataset file')
    p.add_argument('--out_dir',    required=True, help='Directory to save decoded MIDIs')
    p.add_argument('--n_samples',  type=int, default=8,
                   help='Number of MIDI files to save (max 10)')
    p.add_argument('--split',      type=str, default='train',
                   choices=['all', 'train', 'val', 'test'])
    p.add_argument('--train_length', type=int, default=384,
                   help='Frame length in subbeats (must match training)')
    p.add_argument('--batch_size', type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = FramedDataset(args.data, args.train_length, args.batch_size,
                            split=args.split)
    loader  = DataLoader(dataset, batch_size=None, num_workers=0)

    print(f'Decoding {args.n_samples} samples from {args.data} (split={args.split})')
    print(f'Frame length: {args.train_length} subbeats = '
          f'{args.train_length * SECONDS_PER_SUBBEAT:.1f}s '
          f'({args.train_length // 16} bars)')

    n_samples = min(args.n_samples, 10)

    for i, batch in enumerate(loader):
        if i >= n_samples:
            break

        # FramedDataset returns [data (B, T, 16), lengths (B,)]
        data = batch[0]          # (B, T, 16)
        frame = data[0]          # (T, 16)
        pm    = decode_frame_to_midi(frame)

        out_path = os.path.join(args.out_dir, f'sample_{i:03d}.mid')
        pm.write(out_path)

        # Print a summary of pitches seen
        notes = [n for inst in pm.instruments for n in inst.notes]
        pitches = sorted({n.pitch % 12 for n in notes})
        names   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        print(f'  sample_{i:03d}.mid  {len(notes):3d} notes  '
              f'pitch classes: {[names[p] for p in pitches]}')

    print(f'\nSaved to {args.out_dir}')


if __name__ == '__main__':
    main()
