"""
Extract bar-level chord token sequences from XF-MIDI files and save them
as a sidecar file alongside an existing preprocessed dataset.

Usage
-----
  python -m midi_adapter.preprocess_chords \\
      --txt   data/rwc_cp16_v2.txt \\
      --root  /path/to/rwc/midi/folder \\
      --out   data/rwc_cp16_v2.bar_chords.pt

The output is a list of 1-D int16 tensors (one per song in the dataset),
where tensor[s][b] is the chord token for bar b of song s.
Index order matches the song order in the .txt file exactly, so
bar_chords[s] aligns with the s-th song in the paired .pt data file.

Songs whose MIDI file cannot be found or parsed get an empty tensor [];
ChordConditionedDataset will fill those bars with NO_CHORD_TOKEN.
"""

from __future__ import annotations

import argparse
import os

import torch

from midi_adapter.chord_tokenizer import chords_to_bar_tokens, NO_CHORD_TOKEN


def _load_txt(txt_path: str) -> list[tuple[int, str]]:
    """Read 'idx TAB relative_path' lines produced by create_npy_dataset_from_midi."""
    entries = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, rel_path = line.split('\t', 1)
            entries.append((int(idx), rel_path))
    return entries


def extract_bar_chords(
    txt_path:         str,
    midi_root:        str,
    length_path:      str,
    beat_div:         int = 4,
    subbeats_per_bar: int = 16,
    verbose:          bool = True,
) -> list[torch.Tensor]:
    """
    For each song listed in txt_path, open the corresponding XF-MIDI file,
    extract chord events, and convert to a bar-level int16 tensor.

    Parameters
    ----------
    txt_path         : path to the .txt index file from create_npy_dataset_from_midi
    midi_root        : root directory that relative MIDI paths are resolved against
    length_path      : path to the .length.pt file (song lengths in subbeats)
    beat_div         : subbeats per beat (must match the value used during preprocessing)
    subbeats_per_bar : subbeats per bar (beat_div * beats_per_bar, typically 16)

    Returns
    -------
    List of 1-D int16 tensors, one per song.  Empty tensor for failed songs.
    """
    try:
        from xf_midi import XFMidi
    except ImportError:
        raise ImportError("xf_midi is required; make sure it is on sys.path")

    entries     = _load_txt(txt_path)
    lengths     = torch.load(length_path, weights_only=True)   # (n_songs,) subbeats
    bar_chords  = []

    for i, (song_idx, rel_path) in enumerate(entries):
        midi_path = os.path.join(midi_root, rel_path)
        n_subbeats = int(lengths[i].item())
        n_bars     = (n_subbeats + subbeats_per_bar - 1) // subbeats_per_bar

        if not os.path.exists(midi_path):
            if verbose:
                print(f'[{i}] MISSING: {midi_path}')
            bar_chords.append(torch.full((n_bars,), NO_CHORD_TOKEN, dtype=torch.int16))
            continue

        try:
            midi = XFMidi(midi_path, constant_tempo=60.0 / beat_div)
            tokens = chords_to_bar_tokens(midi.chords, n_subbeats, subbeats_per_bar)
            bar_chords.append(torch.tensor(tokens, dtype=torch.int16))
        except Exception as e:
            if verbose:
                print(f'[{i}] ERROR {midi_path}: {e}')
            bar_chords.append(torch.full((n_bars,), NO_CHORD_TOKEN, dtype=torch.int16))

        if verbose and (i + 1) % 50 == 0:
            print(f'Processed {i + 1}/{len(entries)} songs')

    return bar_chords


def main():
    p = argparse.ArgumentParser(description='Extract bar-level chord tokens from XF-MIDI dataset')
    p.add_argument('--txt',    required=True, help='Path to dataset .txt index file')
    p.add_argument('--root',   required=True, help='Root folder for MIDI files')
    p.add_argument('--length', required=True, help='Path to dataset .length.pt file')
    p.add_argument('--out',    required=True, help='Output path for bar_chords.pt')
    p.add_argument('--beat_div',          type=int, default=4)
    p.add_argument('--subbeats_per_bar',  type=int, default=16)
    args = p.parse_args()

    bar_chords = extract_bar_chords(
        txt_path         = args.txt,
        midi_root        = args.root,
        length_path      = args.length,
        beat_div         = args.beat_div,
        subbeats_per_bar = args.subbeats_per_bar,
    )
    torch.save(bar_chords, args.out)
    print(f'Saved {len(bar_chords)} songs → {args.out}')


if __name__ == '__main__':
    main()
