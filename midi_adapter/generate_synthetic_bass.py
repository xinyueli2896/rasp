"""
Generate synthetic bass MIDI dataset with paired bar-level chord annotations.

The chord progression follows the EXACT same rule as the RASP experiment:

    RASP:  token[i]        = (x + OFFSETS[i % 4]) % 12
    Bass:  chord_root[bar] = (key + OFFSETS[bar % 4]) % 12
           OFFSETS = [0, 5, 7, 0]   (period-4: I – IV – V – I)

Example (key = C = 0):  C, F, G, C, C, F, G, C, ...
Example (key = D = 2):  D, G, A, D, D, G, A, D, ...

  - key  = RASP starter x
  - OFFSETS = [0,5,7,0] = the fixed rule (same as RASP)
  - bar  = sequence position i
  - 12   = modulus N

Two modes
---------
  monophonic (default):
    Single bass instrument (program=33). One note per beat: the chord root
    in octave 2 (C2=36..B2=47). Follows the I-IV-V-I rule exactly.
    Simple and rule-transparent, but far from the CP transformer's polyphonic
    training distribution.

  polyphonic (--polyphonic):
    Single piano instrument (program=0). Four voices per beat, placed in
    octaves 2-5, all chord tones of the expected chord:
        voice 0 (bass):    root     in oct 2  (C2=36..B2=47)
        voice 1 (tenor):   interval 1 in oct 3  (C3=48..B3=59)
        voice 2 (alto):    interval 2 in oct 4  (C4=60..B4=71)
        voice 3 (soprano): interval 3 in oct 5  (C5=72..B5=83)
    For triads (3 tones), voice 3 doubles voice 0's pitch class one octave up.
    Notes are added ascending by pitch so voice 0 is always the bass note.
    Use --polyphonic to match the CP transformer's actual polyphonic training
    distribution for adapter training.

Outputs
-------
  <out_dir>/*.mid                     individual MIDI files
  <out_pt>.pt                         CP tensor data (all songs concatenated)
  <out_pt>.length.pt                  per-song lengths in subbeats
  <out_pt>.pitch_shift_range.pt       per-song pitch shift range (int8)
  <out_pt>.beat_chords.pt             beat-level chord tokens (list of int16 tensors)
  <out_pt>.txt                        song index → MIDI filename mapping

Recommended workflow
--------------------
  # Polyphonic adapter training (matches CP transformer distribution)
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 3000 --n_bars 128 --polyphonic \\
      --out_dir data/bass_poly_pretrain --out_pt data/bass_poly_pretrain_cp4 \\
      --keys 0 2 4 5 7 9 11

  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 5000 --n_bars 128 --polyphonic \\
      --out_dir data/bass_poly_all --out_pt data/bass_poly_all_cp4

  # Original monophonic (kept for reference/ablation)
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 3000 --n_bars 128 \\
      --out_dir data/bass_pretrain --out_pt data/bass_pretrain_cp4 \\
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

BEAT_DIV           = 1                              # 1 subbeat = 1 beat (quarter note)
BEATS_PER_BAR      = 4                              # 4/4 time
SUBBEATS_PER_BAR   = BEAT_DIV * BEATS_PER_BAR      # 4
CONSTANT_TEMPO     = 120.0                          # BPM
SECONDS_PER_SUBBEAT = 60.0 / CONSTANT_TEMPO / BEAT_DIV  # 0.5 s/subbeat (= 1 beat)

DURATION_TEMPLATES = np.array([
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
    192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
])
DURATION_BOUNDARIES = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2.0

# Bass range: one octave only (C2=36 … B2=47) so every pitch class maps to
# exactly one MIDI note — no octave ambiguity during training or evaluation.
BASS_MIN, BASS_MAX = 36, 47

# Octave bases for voiced-chord voices (C of each octave)
_VOICE_OCTAVE_BASES = [36, 48, 60, 72]   # C2, C3, C4, C5

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# (quality_name, chord_tone_intervals_above_root)
COMMON_CHORDS: dict[str, list[int]] = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    '7':    [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
}

# Exactly the RASP rule offsets: token[i] = (x + OFFSETS[i%4]) % 12
# Period-4 pattern: I – IV – V – I
OFFSETS = [0, 5, 7, 0]


# ---------------------------------------------------------------------------
# Note generation helpers
# ---------------------------------------------------------------------------

def _bass_root(root_semitone: int) -> int:
    """MIDI pitch of the bass root, always in octave 2 (C2=36 … B2=47).
    All 12 semitones land in the same register so the I-IV-V-I cycle is
    perceived as a consistent pitch sequence, not register jumps."""
    return 36 + root_semitone   # C2=36, D2=38, …, B2=47


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
        start=float(start) * SECONDS_PER_SUBBEAT,
        end=float(start + duration_subbeats) * SECONDS_PER_SUBBEAT,
    )


def _voiced_chord_pitches(root_semitone: int, quality: str) -> list[int]:
    """MIDI pitches for a 4-voice chord spread across octaves 2-5.

    Voice 0 (bass):    root               in C2 octave (36+pc)
    Voice 1 (tenor):   intervals[1]       in C3 octave (48+pc)
    Voice 2 (alto):    intervals[2]       in C4 octave (60+pc)
    Voice 3 (soprano): intervals[3 or 0]  in C5 octave (72+pc)

    For triads (3 tones), voice 3 doubles voice 0's pitch class an octave higher
    so the chromagram matches the expected 3-tone chord.
    Notes are already in ascending pitch order (bass first).
    """
    intervals = COMMON_CHORDS[quality]   # 3 or 4 intervals
    pitches = []
    for v, base in enumerate(_VOICE_OCTAVE_BASES):
        interval = intervals[v % len(intervals)]
        pitches.append(base + (root_semitone + interval) % 12)
    return pitches   # [bass, tenor, alto, soprano]  ascending pitch order


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
        step = T // 4
        # I-IV-V-I within each bar: root_semitone is the song key
        pitches = [_bass_root((root_semitone + offset) % 12) for offset in OFFSETS]
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
    n_bars:                  int        = 32,
    key:                     int | None = None,   # root semitone (= RASP starter x)
    allowed_keys:            list[int]  | None = None,
    quality:                 str | None = None,   # None = random once per song
    bass_instrument_program: int        = 33,     # Electric Bass, finger
    polyphonic:              bool       = False,
) -> tuple[pretty_midi.PrettyMIDI, list[list]]:
    """
    Generate one song following the exact RASP rule:

        chord_root[beat] = (key + OFFSETS[beat % 4]) % 12
        OFFSETS = [0, 5, 7, 0]

    monophonic (polyphonic=False):
        Single bass instrument (program=bass_instrument_program).
        One note per beat: the chord root in octave 2 (C2=36..B2=47).

    polyphonic (polyphonic=True):
        Single piano instrument (program=0). Four voices per beat:
            voice 0 (bass):    chord root in oct 2 (36+pc)
            voice 1 (tenor):   chord tone in oct 3 (48+pc)
            voice 2 (alto):    chord tone in oct 4 (60+pc)
            voice 3 (soprano): chord tone in oct 5 (72+pc)
        All notes added ascending by pitch so voice 0 is the bass.
        Use this mode for adapter training to match the CP transformer's
        polyphonic training distribution.

    Returns
    -------
    pm        : PrettyMIDI object
    xf_chords : [[subbeat_time_float, chord_str], ...]
    """
    if allowed_keys is None:
        allowed_keys = list(range(12))
    if key is None:
        key = random.choice(allowed_keys)

    song_quality = quality if quality is not None else random.choice(list(COMMON_CHORDS.keys()))

    pm   = pretty_midi.PrettyMIDI(initial_tempo=CONSTANT_TEMPO)
    pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]

    if polyphonic:
        inst = pretty_midi.Instrument(program=0, name='Piano')
    else:
        inst = pretty_midi.Instrument(program=bass_instrument_program, name='Bass')

    xf_chords: list[list] = []
    beat_step = SUBBEATS_PER_BAR // BEATS_PER_BAR   # 1 subbeat per beat (BEAT_DIV=1)

    for b in range(n_bars):
        bar_start = b * SUBBEATS_PER_BAR
        for j, offset in enumerate(OFFSETS):
            beat_root  = (key + offset) % 12
            beat_start = bar_start + j * beat_step
            chord_str  = f'{ROOT_NAMES[beat_root]}:{song_quality}'
            xf_chords.append([float(beat_start), chord_str])

            if polyphonic:
                # 4 voices: bass (oct2), tenor (oct3), alto (oct4), soprano (oct5)
                # Added in ascending pitch order so _preprocess_pm assigns voice 0 = bass.
                pitches = _voiced_chord_pitches(beat_root, song_quality)
                vel_base = 70
                for v_idx, pitch in enumerate(pitches):
                    vel = max(40, min(100, vel_base - v_idx * 5 + random.randint(-5, 5)))
                    inst.notes.append(_make_note(pitch, beat_start, beat_step, vel))
            else:
                inst.notes.append(
                    _make_note(_bass_root(beat_root), beat_start, beat_step)
                )

    pm.instruments.append(inst)
    return pm, xf_chords


# ---------------------------------------------------------------------------
# Preprocessing  (no xf_midi dependency needed for synthetic data)
# ---------------------------------------------------------------------------

def pitch_sort_cp(data: torch.Tensor, tuple_size: int = 4) -> torch.Tensor:
    """Sort voices at each timestep by ascending pitch (lowest first).

    Required for Approach 1 (bass note regulation) when using polyphonic CP data:
    ensures voice 0 always contains the lowest-pitched note so that
    x_proc[:, :, 1] % 128 % 12 gives the bass pitch class.

    data : (n_subbeats, max_polyphony * tuple_size)  raw uint8 CP tensor
           Each voice v occupies slots [v*tuple_size : (v+1)*tuple_size]
           = [prog, pitch, dur, vel].  Padding = 255; EOS prog = 254.

    Returns a tensor of the same shape with voices sorted by pitch ascending,
    padding and EOS entries sorted to the end.
    """
    n_sub, flat = data.shape
    n_voices = flat // tuple_size
    d = data.view(n_sub, n_voices, tuple_size)  # (n, v, 4)

    pitch = d[:, :, 1].long()   # (n, v)  — 255 for pad, real pitch for notes
    # Sort key: pitch ascending; padding (255) sorts last because 255 > all real pitches
    sort_idx = pitch.argsort(dim=1, stable=True)   # (n, v)
    sorted_d = d.gather(1, sort_idx.unsqueeze(-1).expand_as(d))
    return sorted_d.reshape(n_sub, flat)


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
            s = int(round(note.start / SECONDS_PER_SUBBEAT))
            e = int(round(note.end   / SECONDS_PER_SUBBEAT))
            if 0 <= s < n_subbeats and polyphony_counts[s] < max_polyphony:
                dur = int(np.searchsorted(DURATION_BOUNDARIES, e - s))
                min_pitch = min(min_pitch, note.pitch)
                max_pitch = max(max_pitch, note.pitch)
                rolls[s, polyphony_counts[s]] = [prog, note.pitch, dur, note.velocity]
                polyphony_counts[s] += 1

    for i in range(n_subbeats):
        if polyphony_counts[i] < max_polyphony:
            rolls[i, polyphony_counts[i], 0] = 254   # EOS token

    ps_min = 0
    ps_max = 0

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
    pitch_sort:    bool      = False,
    polyphonic:    bool      = False,
    quality:       str | None = None,
) -> None:
    """
    Generate n_songs synthetic songs and save all output files.

    Parameters
    ----------
    out_dir    : folder where individual .mid files are written
    out_pt     : prefix for the dataset files (e.g. 'data/synthetic_bass_cp4')
    polyphonic : if True, generate 4-voice piano chords instead of monophonic bass;
                 use this for adapter training to match the CP transformer distribution.
    pitch_sort : sort voices by ascending pitch at each timestep (no-op in polyphonic
                 mode since notes are already added ascending, but safe to enable both).
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
            polyphonic   = polyphonic,
            quality      = quality,
        )

        # Save MIDI file
        prefix = 'poly' if polyphonic else 'bass'
        rel_path  = f'{prefix}_{i:06d}.mid'
        midi_path = os.path.join(out_dir, rel_path)
        pm.write(midi_path)
        txt_lines.append(f'{i}\t{rel_path}')

        # Preprocess to CP tensor (optionally sort voices by pitch ascending)
        data, shift = _preprocess_pm(pm, n_subbeats, max_polyphony)
        if pitch_sort or polyphonic:
            # polyphonic notes are already added ascending, but pitch_sort is safe
            # to apply as a correctness guarantee
            data = pitch_sort_cp(data)
        all_data.append(data)
        all_shift.append(shift)

        # Convert chord annotations to beat-level tokens (one per beat)
        from midi_adapter.chord_tokenizer import chords_to_beat_tokens
        beat_tokens = chords_to_beat_tokens(xf_chords, n_subbeats)
        all_chords.append(torch.tensor(beat_tokens, dtype=torch.int16))

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{n_songs} songs generated')

    # Save in FramedDataset-compatible format
    torch.save(torch.cat(all_data, dim=0),     f'{out_pt}.pt')
    torch.save(torch.tensor([n_subbeats] * n_songs), f'{out_pt}.length.pt')
    torch.save(torch.stack(all_shift, dim=0),  f'{out_pt}.pitch_shift_range.pt')
    torch.save(all_chords,                     f'{out_pt}.beat_chords.pt')

    with open(f'{out_pt}.txt', 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    print(f'Saved {n_songs} songs to {out_pt}.{{pt,length.pt,pitch_shift_range.pt,beat_chords.pt,txt}}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Generate synthetic bass MIDI dataset')
    p.add_argument('--n_songs',       type=int,   default=5000)
    p.add_argument('--n_bars',        type=int,   default=128,
                   help='Bars per song (must be >= TRAIN_LENGTH/SUBBEATS_PER_BAR=96 for training)')
    p.add_argument('--out_dir',       type=str,   required=True,
                   help='Directory for individual MIDI files')
    p.add_argument('--out_pt',        type=str,   required=True,
                   help='Output prefix for .pt dataset files')
    p.add_argument('--max_polyphony', type=int,   default=4)
    p.add_argument('--keys',          type=int,   nargs='*', default=None,
                   help='Allowed root semitones (0-11). Default: all 12.')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--pitch_sort',    action='store_true',
                   help='Sort voices at each timestep by ascending pitch so voice 0 '
                        'always holds the lowest note. Applied automatically when '
                        '--polyphonic is set.')
    p.add_argument('--polyphonic',    action='store_true',
                   help='Generate 4-voice piano chords instead of monophonic bass. '
                        'All voices play chord tones across octaves 2-5 so the CP '
                        'tensor matches the CP transformer\'s polyphonic training '
                        'distribution. Strongly recommended for adapter training.')
    p.add_argument('--quality', type=str, default=None,
                   choices=list(COMMON_CHORDS.keys()),
                   help='Fix chord quality for all songs (default: random per song). '
                        'Use "maj" for the chord approach to stay consistent with '
                        'CPChordRuleModel which uses major triad intervals (0,4,7).')
    args = p.parse_args()

    print(f'Generating {args.n_songs} songs, {args.n_bars} bars each')
    print(f'Rule: chord_root[bar] = (key + OFFSETS[bar%4]) % 12  OFFSETS={OFFSETS}')
    print(f'Keys: {args.keys if args.keys else "all 12"}')

    generate_dataset(
        n_songs       = args.n_songs,
        out_dir       = args.out_dir,
        out_pt        = args.out_pt,
        n_bars        = args.n_bars,
        max_polyphony = args.max_polyphony,
        allowed_keys  = args.keys,
        seed          = args.seed,
        pitch_sort    = args.pitch_sort,
        polyphonic    = args.polyphonic,
        quality       = args.quality,
    )


if __name__ == '__main__':
    main()
