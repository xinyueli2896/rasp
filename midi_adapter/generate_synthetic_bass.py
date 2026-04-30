"""
Generate synthetic bass MIDI dataset with paired bar-level chord annotations.

Each song is a random chord progression with bass notes following the chords.
All output files are directly compatible with the existing preprocessing
pipeline (FramedDataset format) and ChordConditionedDataset.

Outputs
-------
  <out_dir>/*.mid                     individual MIDI files
  <out_pt>.pt                         CP tensor data (all songs concatenated)
  <out_pt>.length.pt                  per-song lengths in subbeats
  <out_pt>.pitch_shift_range.pt       per-song pitch shift range (int8)
  <out_pt>.bar_chords.pt              bar-level chord tokens (list of int16 tensors)
  <out_pt>.txt                        song index → MIDI filename mapping

RASP-analogy note
-----------------
Use --keys to restrict the pretrain set (e.g. --keys 0 2 4 5 7 9 11 for
the white keys of C major) and train the CP transformer on that subset only.
Then use all 12 keys for adapter training.  The adapter must then generalise
chord conditioning to the 5 unseen keys, analogous to the RASP generalization
experiment.

Usage
-----
  # Full dataset (all keys, all chord types)
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 5000 --n_bars 32 --out_dir data/synthetic_bass \\
      --out_pt  data/synthetic_bass_cp4

  # Pretrain-only subset (7 keys)
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 3000 --n_bars 32 --out_dir data/synthetic_bass_pretrain \\
      --out_pt  data/synthetic_bass_pretrain_cp4 \\
      --keys 0 2 4 5 7 9 11
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

try:
    import pretty_midi
except ImportError:
    print('pretty_midi is required: pip install pretty_midi')
    sys.exit(1)

from midi_adapter.chord_tokenizer import (
    chord_str_to_token, CHORD_MAP, N_CHORD_TOKENS, NO_CHORD_TOKEN,
)

# ---------------------------------------------------------------------------
# Constants  (must match cp_transformer.py / preprocess_large_midi_dataset.py)
# ---------------------------------------------------------------------------

BEAT_DIV         = 4                           # subbeats per beat
BEATS_PER_BAR    = 4                           # 4/4 time
SUBBEATS_PER_BAR = BEAT_DIV * BEATS_PER_BAR   # 16
CONSTANT_TEMPO   = 60.0 / BEAT_DIV             # 15.0 BPM → 1 subbeat = 1 second

DURATION_TEMPLATES = np.array([
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
    192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
])
DURATION_BOUNDARIES = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2.0

# Bass range: E1 (MIDI 28) to B3 (MIDI 59)
BASS_MIN, BASS_MAX = 28, 59

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# (quality_name, chord_tone_intervals_above_root)
COMMON_CHORDS: dict[str, list[int]] = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    '7':    [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
}

# Typical chord scale degrees (intervals above key root) for common progressions
SCALE_DEGREES = [0, 5, 7, 9]   # I, IV, V, vi


# ---------------------------------------------------------------------------
# Note generation helpers
# ---------------------------------------------------------------------------

def _bass_root(root_semitone: int) -> int:
    """MIDI pitch of the bass root in the bass register."""
    pitch = 24 + root_semitone   # start at C1
    while pitch < BASS_MIN:
        pitch += 12
    while pitch > BASS_MAX:
        pitch -= 12
    return pitch


def _chord_tones(root_semitone: int, quality: str) -> list[int]:
    root = _bass_root(root_semitone)
    tones = [root + i for i in COMMON_CHORDS[quality]]
    # keep in range; allow one octave up if needed
    tones = [(t if t <= BASS_MAX else t - 12) for t in tones]
    tones = [(t if t >= BASS_MIN else t + 12) for t in tones]
    return tones


def _make_note(pitch, start, duration_subbeats, velocity=70):
    return pretty_midi.Note(
        velocity=velocity,
        pitch=int(pitch),
        start=float(start),
        end=float(start + duration_subbeats),
    )


def _bar_notes(root_semitone: int, quality: str,
               bar_start: int, pattern: str) -> list[pretty_midi.Note]:
    """Generate bass notes for one bar."""
    tones = _chord_tones(root_semitone, quality)
    root  = tones[0]
    fifth = tones[min(2, len(tones) - 1)]

    T = SUBBEATS_PER_BAR   # 16

    if pattern == 'whole':
        return [_make_note(root, bar_start, T)]

    if pattern == 'half':
        return [
            _make_note(root,  bar_start,          T // 2),
            _make_note(fifth, bar_start + T // 2, T // 2),
        ]

    if pattern == 'quarter':
        pitches = [root, fifth, root, fifth]
        step    = T // 4
        return [_make_note(p, bar_start + i * step, step) for i, p in enumerate(pitches)]

    if pattern == 'walking':
        # Walk through chord tones across 4 beats
        step    = T // 4
        pitches = [tones[i % len(tones)] for i in range(4)]
        return [_make_note(p, bar_start + i * step, step) for i, p in enumerate(pitches)]

    raise ValueError(f'Unknown pattern: {pattern}')


# ---------------------------------------------------------------------------
# Song generation
# ---------------------------------------------------------------------------

PATTERNS = ['whole', 'half', 'quarter', 'walking']


def generate_song(
    n_bars:                 int        = 32,
    key:                    int | None = None,    # root semitone, None = random
    allowed_keys:           list[int]  | None = None,
    allowed_qualities:      list[str]  | None = None,
    bars_per_chord:         int        = 2,
    bass_instrument_program: int       = 33,      # Electric Bass, finger
) -> tuple[pretty_midi.PrettyMIDI, list[list]]:
    """
    Returns
    -------
    pm        : PrettyMIDI object
    xf_chords : [[subbeat_time_float, chord_str], ...] (compatible with
                chord_tokenizer.chords_to_bar_tokens)
    """
    if allowed_keys is None:
        allowed_keys = list(range(12))
    if allowed_qualities is None:
        allowed_qualities = list(COMMON_CHORDS.keys())

    if key is None:
        key = random.choice(allowed_keys)

    # Build a 4-chord progression
    degrees   = random.sample(SCALE_DEGREES, min(4, len(SCALE_DEGREES)))
    quality_choice = random.choices(allowed_qualities, k=len(degrees))
    progression = [((key + d) % 12, q) for d, q in zip(degrees, quality_choice)]

    # Assign one chord per bar (repeat progression)
    chord_per_bar = [
        progression[(b // bars_per_chord) % len(progression)]
        for b in range(n_bars)
    ]

    pm   = pretty_midi.PrettyMIDI(initial_tempo=CONSTANT_TEMPO)
    bass = pretty_midi.Instrument(program=bass_instrument_program, name='Bass')

    xf_chords: list[list] = []

    for b, (root, quality) in enumerate(chord_per_bar):
        bar_start = b * SUBBEATS_PER_BAR
        chord_str = f'{ROOT_NAMES[root]}:{quality}'
        xf_chords.append([float(bar_start), chord_str])

        pattern = random.choice(PATTERNS)
        bass.notes.extend(_bar_notes(root, quality, bar_start, pattern))

    pm.instruments.append(bass)
    return pm, xf_chords


# ---------------------------------------------------------------------------
# Preprocessing  (no xf_midi dependency needed for synthetic data)
# ---------------------------------------------------------------------------

def _preprocess_pm(
    pm:            pretty_midi.PrettyMIDI,
    n_subbeats:    int,
    max_polyphony: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a PrettyMIDI object to CP tensor format.
    Equivalent to preprocess_midi(...) from preprocess_large_midi_dataset.py
    but uses pretty_midi directly (no xf_midi required).

    Returns
    -------
    data        : (n_subbeats, max_polyphony * 4)  uint8 tensor
    pitch_range : (2,)                             int8  tensor [min_shift, max_shift]
    """
    rolls           = np.full((n_subbeats, max_polyphony, 4), 255, dtype=np.uint8)
    polyphony_counts = np.zeros(n_subbeats, dtype=np.uint8)
    min_pitch, max_pitch = 127, 0

    for inst in pm.instruments:
        prog = inst.program
        for note in inst.notes:
            # note.start is in seconds; at CONSTANT_TEMPO=15 BPM, 1 s = 1 subbeat
            s = int(round(note.start))
            e = int(round(note.end))
            if 0 <= s < n_subbeats and polyphony_counts[s] < max_polyphony:
                dur = int(np.searchsorted(DURATION_BOUNDARIES, e - s))
                min_pitch = min(min_pitch, note.pitch)
                max_pitch = max(max_pitch, note.pitch)
                rolls[s, polyphony_counts[s]] = [prog, note.pitch, dur, note.velocity]
                polyphony_counts[s] += 1

    for i in range(n_subbeats):
        if polyphony_counts[i] < max_polyphony:
            rolls[i, polyphony_counts[i], 0] = 254   # EOS token

    ps_min = int(np.clip(-min_pitch,       -127, 127))
    ps_max = int(np.clip(127 - max_pitch,  -127, 127))
    # clamp pitch shift to [-5, 6] as in FramedDataset
    ps_min = max(ps_min, -5)
    ps_max = min(ps_max,  6)

    data = torch.tensor(rolls.reshape(n_subbeats, max_polyphony * 4), dtype=torch.uint8)
    return data, torch.tensor([ps_min, ps_max], dtype=torch.int8)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_songs:       int,
    out_dir:       str,
    out_pt:        str,
    n_bars:        int       = 32,
    max_polyphony: int       = 4,
    allowed_keys:  list[int] | None = None,
    seed:          int       = 42,
) -> None:
    """
    Generate n_songs synthetic bass songs and save all output files.

    Parameters
    ----------
    out_dir  : folder where individual .mid files are written
    out_pt   : prefix for the dataset files (e.g. 'data/synthetic_bass_cp4')
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    all_data:   list[torch.Tensor] = []
    all_shift:  list[torch.Tensor] = []
    all_chords: list[torch.Tensor] = []
    txt_lines:  list[str]          = []

    n_subbeats = n_bars * SUBBEATS_PER_BAR

    for i in range(n_songs):
        pm, xf_chords = generate_song(
            n_bars       = n_bars,
            allowed_keys = allowed_keys,
        )

        # Save MIDI file
        rel_path = f'bass_{i:06d}.mid'
        midi_path = os.path.join(out_dir, rel_path)
        pm.write(midi_path)
        txt_lines.append(f'{i}\t{rel_path}')

        # Preprocess to CP tensor
        data, shift = _preprocess_pm(pm, n_subbeats, max_polyphony)
        all_data.append(data)
        all_shift.append(shift)

        # Convert chord annotations to bar tokens
        from midi_adapter.chord_tokenizer import chords_to_bar_tokens
        bar_tokens = chords_to_bar_tokens(xf_chords, n_subbeats, SUBBEATS_PER_BAR)
        all_chords.append(torch.tensor(bar_tokens, dtype=torch.int16))

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{n_songs} songs generated')

    # Save in FramedDataset-compatible format
    torch.save(torch.cat(all_data, dim=0),     f'{out_pt}.pt')
    torch.save(torch.tensor([n_subbeats] * n_songs), f'{out_pt}.length.pt')
    torch.save(torch.stack(all_shift, dim=0),  f'{out_pt}.pitch_shift_range.pt')
    torch.save(all_chords,                     f'{out_pt}.bar_chords.pt')

    with open(f'{out_pt}.txt', 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    print(f'Saved {n_songs} songs to {out_pt}.{{pt,length.pt,pitch_shift_range.pt,bar_chords.pt,txt}}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Generate synthetic bass MIDI dataset')
    p.add_argument('--n_songs',       type=int,   default=5000)
    p.add_argument('--n_bars',        type=int,   default=32,
                   help='Bars per song (must be >= target_length/16 for training)')
    p.add_argument('--out_dir',       type=str,   required=True,
                   help='Directory for individual MIDI files')
    p.add_argument('--out_pt',        type=str,   required=True,
                   help='Output prefix for .pt dataset files')
    p.add_argument('--max_polyphony', type=int,   default=4)
    p.add_argument('--keys',          type=int,   nargs='*', default=None,
                   help='Allowed root semitones (0-11). Default: all 12.')
    p.add_argument('--seed',          type=int,   default=42)
    args = p.parse_args()

    print(f'Generating {args.n_songs} songs, {args.n_bars} bars each')
    print(f'Keys: {args.keys if args.keys else "all 12"}')

    generate_dataset(
        n_songs       = args.n_songs,
        out_dir       = args.out_dir,
        out_pt        = args.out_pt,
        n_bars        = args.n_bars,
        max_polyphony = args.max_polyphony,
        allowed_keys  = args.keys,
        seed          = args.seed,
    )


if __name__ == '__main__':
    main()
