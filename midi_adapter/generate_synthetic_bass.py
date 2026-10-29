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

Base CP transformer is trained only on seen keys → learns the I-IV-V-I cycle.
Adapter (chord-conditioned) must generalise that cycle to unseen keys —
exactly as the RASP adapter must generalise to unseen starters.

Outputs
-------
  <out_dir>/*.mid                     individual MIDI files
  <out_pt>.pt                         CP tensor data (all songs concatenated)
  <out_pt>.length.pt                  per-song lengths in subbeats
  <out_pt>.pitch_shift_range.pt       per-song pitch shift range (int8)
  <out_pt>.beat_chords.pt             beat-level chord tokens (list of int16 tensors)
  <out_pt>.txt                        song index → MIDI filename mapping

RASP-analogy workflow
---------------------
  # Pretrain CP transformer on seen keys (white keys = starters {0,2,4,5,7,9,11})
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 3000 --n_bars 32 \\
      --out_dir data/bass_pretrain --out_pt data/bass_pretrain_cp4 \\
      --keys 0 2 4 5 7 9 11

  # Full dataset (all 12 keys) for adapter fine-tuning
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 5000 --n_bars 32 \\
      --out_dir data/bass_all --out_pt data/bass_all_cp4
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
# v3 octave variation may place a note an octave either side of that band.
OCTAVE_MIN, OCTAVE_MAX = 24, 59          # C1 .. B3

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# (quality_name, chord_tone_intervals_above_root)
COMMON_CHORDS: dict[str, list[int]] = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    '7':    [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
}

# Parent scale per quality, for picking non-chord tones that still belong to
# the harmony's key rather than arbitrary semitones.
SCALES: dict[str, list[int]] = {
    'maj':  [0, 2, 4, 5, 7, 9, 11],
    'min':  [0, 2, 3, 5, 7, 8, 10],
    '7':    [0, 2, 4, 5, 7, 9, 10],      # mixolydian
    'maj7': [0, 2, 4, 5, 7, 9, 11],
    'min7': [0, 2, 3, 5, 7, 8, 10],
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


def _non_chord_tones(root_semitone: int, quality: str) -> list[int]:
    """Scale tones of the bar's harmony that are NOT chord tones."""
    chord = set(COMMON_CHORDS[quality])
    root = _bass_root(root_semitone)
    out = []
    for i in SCALES[quality]:
        if i in chord:
            continue
        t = root + i
        t = t - 12 if t > BASS_MAX else t
        t = t + 12 if t < BASS_MIN else t
        out.append(t)
    return out


def _vary_bar(notes, bar_start: int, root_semitone: int, quality: str,
              octave_prob: float, nct_prob: float) -> None:
    """Apply v3 variation in place: octave displacement and non-chord tones.

    The DOWNBEAT keeps the root's pitch class, always. Evaluation reads the
    key from beat 0 and scores each bar's downbeat against
    (key + OFFSETS[p]) % 12, so a non-chord tone there would break both the
    key extraction and the metric. An octave displacement is safe anywhere --
    it does not change the pitch class -- so the downbeat may move register.
    """
    if not (octave_prob or nct_prob):
        return
    t0 = bar_start * SECONDS_PER_SUBBEAT
    ncts = _non_chord_tones(root_semitone, quality)
    for n in notes:
        on_downbeat = abs(n.start - t0) < 1e-9
        if ncts and not on_downbeat and random.random() < nct_prob:
            n.pitch = int(random.choice(ncts))
        if random.random() < octave_prob:
            shifted = n.pitch + random.choice((-12, 12))
            if OCTAVE_MIN <= shifted <= OCTAVE_MAX:
                n.pitch = int(shifted)


def _make_note(pitch, start, duration_subbeats, velocity=70):
    return pretty_midi.Note(
        velocity=velocity,
        pitch=int(pitch),
        start=float(start) * SECONDS_PER_SUBBEAT,
        end=float(start + duration_subbeats) * SECONDS_PER_SUBBEAT,
    )


def _bar_notes(root_semitone: int, quality: str,
               bar_start: int, pattern: str) -> list[pretty_midi.Note]:
    """Generate bass notes for one bar."""
    tones = _chord_tones(root_semitone, quality)
    root  = tones[0]
    fifth = tones[min(2, len(tones) - 1)]

    T = SUBBEATS_PER_BAR   # 4 -- BEAT_DIV is 1, so one subbeat IS one beat
                           # (quarter note) and a bar holds 4 of them. Nothing
                           # in this data is subdivided below a beat. Note
                           # chord_dataset.py defines its own SUBBEATS_PER_BAR
                           # = 16; that constant belongs to a different
                           # representation and does not apply here.

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

    if pattern == 'root4':
        # The bar's root, four times -- 'C C C C'. Distinct from 'quarter',
        # which puts a whole I-IV-V-I cycle inside one bar.
        step = T // 4
        return [_make_note(root, bar_start + i * step, step) for i in range(4)]

    if pattern == 'root2':
        # Root held for two beats, twice.
        step = T // 2
        return [_make_note(root, bar_start + i * step, step) for i in range(2)]

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
# Patterns that realise ONE chord across a bar. Every one begins on the root,
# which is what keeps the downbeat a clean target for the rule.
BAR_PATTERNS = ['root4', 'root2', 'whole', 'half', 'walking']
# Not a realisation: asks for an independent draw per bar.
MIXED_PATTERN = 'mixed'


def generate_song(
    n_bars:                  int        = 32,
    key:                     int | None = None,   # root semitone (= RASP starter x)
    allowed_keys:            list[int]  | None = None,
    quality:                 str | None = None,   # None = random once per song
    bass_instrument_program: int        = 33,     # Electric Bass, finger
    cycle_unit:              str        = 'beat',
    pattern:                 str | None = None,   # None = random once per song
    octave_prob:             float      = 0.0,    # v3: octave displacement
    nct_prob:                float      = 0.0,    # v3: non-chord tones
) -> tuple[pretty_midi.PrettyMIDI, list[list]]:
    """
    Generate one song following the exact RASP rule:

        chord_root[bar] = (key + OFFSETS[bar % 4]) % 12
        OFFSETS = [0, 5, 7, 0]

    Identical to: token[i] = (x + OFFSETS[i % 4]) % 12

    Quality is fixed for the whole song. Bass pattern is random per bar.

    Returns
    -------
    pm        : PrettyMIDI object
    xf_chords : [[subbeat_time_float, chord_str], ...] (compatible with
                chord_tokenizer.chords_to_bar_tokens)
    """
    if allowed_keys is None:
        allowed_keys = list(range(12))
    if key is None:
        key = random.choice(allowed_keys)

    song_quality = quality if quality is not None else random.choice(list(COMMON_CHORDS.keys()))
    # One pattern for the whole song, like the quality. A training window is
    # drawn from a single song, so this keeps the realisation consistent within
    # a sample: the model sees one way of playing a bar and has to apply it to
    # whatever root the rule supplies, rather than a new pattern every bar.
    # 'mixed' redraws per bar, which is what the data generated before commit
    # 1d06d93 looks like. Models trained on that data have never seen several
    # consecutive bars share a realisation, so a multi-bar prompt built from a
    # single pattern is out of distribution for them.
    song_pattern = pattern if pattern is not None else random.choice(BAR_PATTERNS)

    pm   = pretty_midi.PrettyMIDI(initial_tempo=CONSTANT_TEMPO)
    pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    bass = pretty_midi.Instrument(program=bass_instrument_program, name='Bass')

    xf_chords: list[list] = []

    beat_step = SUBBEATS_PER_BAR // BEATS_PER_BAR  # 4 subbeats per beat

    if cycle_unit not in ('beat', 'bar'):
        raise ValueError(f"cycle_unit must be 'beat' or 'bar', got {cycle_unit!r}")
    if nct_prob and cycle_unit == 'beat':
        # Per-beat data scores EVERY beat against the rule, so no beat is free
        # to carry a non-chord tone. Octave variation is still fine there.
        raise ValueError('nct_prob requires --cycle_unit bar; in beat mode '
                         'every beat is a scored downbeat')

    for b in range(n_bars):
        bar_start = b * SUBBEATS_PER_BAR
        if cycle_unit == 'beat':
            # One cycle per bar: the root changes every BEAT, so a bar holds a
            # whole I-IV-V-I and every note follows from the key.
            for j, offset in enumerate(OFFSETS):
                beat_root  = (key + offset) % 12
                beat_start = bar_start + j * beat_step
                chord_str  = f'{ROOT_NAMES[beat_root]}:{song_quality}'
                xf_chords.append([float(beat_start), chord_str])
            _beat_notes = _bar_notes(key, song_quality, bar_start, 'quarter')
            _vary_bar(_beat_notes, bar_start, key, song_quality, octave_prob, 0.0)
            bass.notes.extend(_beat_notes)
        else:
            # One chord per BAR: the root changes every bar, so a cycle spans
            # four bars (16 beats). The bar is realised by a pattern which may
            # use chord tones, so the rule fixes the bar's HARMONY while the
            # surface notes are free. Every pattern starts on the root, which
            # keeps the downbeat the clean target.
            bar_root = (key + OFFSETS[b % 4]) % 12
            xf_chords.append([float(bar_start),
                              f'{ROOT_NAMES[bar_root]}:{song_quality}'])
            bar_pattern = (random.choice(BAR_PATTERNS)
                           if song_pattern == MIXED_PATTERN else song_pattern)
            _bn = _bar_notes(bar_root, song_quality, bar_start, bar_pattern)
            _vary_bar(_bn, bar_start, bar_root, song_quality,
                      octave_prob, nct_prob)
            bass.notes.extend(_bn)

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
    octave_prob:   float = 0.0,
    nct_prob:      float = 0.0,
    seed:          int       = 42,
    write_midi:    bool      = True,
    cycle_unit:    str       = 'beat',
    pattern:       str | None = None,
) -> None:
    """
    Generate n_songs synthetic bass songs and save all output files.

    Parameters
    ----------
    out_dir  : folder where individual .mid files are written
    out_pt   : prefix for the dataset files (e.g. 'data/synthetic_bass_cp4')
    write_midi : write the per-song .mid files. Training never reads them --
        the CP tensors are built from the in-memory PrettyMIDI object below,
        and the loaders open only <out_pt>.pt / .length.pt /
        .pitch_shift_range.pt. At a few thousand songs the .mid files are
        usually the largest thing on disk, so pass False unless you want to
        listen to them.
    """
    random.seed(seed)
    np.random.seed(seed)
    if write_midi:
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
            octave_prob  = octave_prob,
            nct_prob     = nct_prob,
            cycle_unit   = cycle_unit,
            pattern      = pattern,
        )

        # Save MIDI file (optional -- nothing downstream reads it)
        rel_path = f'bass_{i:06d}.mid'
        if write_midi:
            pm.write(os.path.join(out_dir, rel_path))
        txt_lines.append(f'{i}\t{rel_path}')

        # Preprocess to CP tensor
        data, shift = _preprocess_pm(pm, n_subbeats, max_polyphony)
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
    p.add_argument('--cycle_unit', type=str, default='beat', choices=['beat','bar'],
                   help="'beat' (default): root changes every beat, one cycle "
                        "per bar, every note determined by the key. 'bar': root "
                        'changes every BAR, a cycle spans 4 bars (16 beats), '
                        'and the bar is realised by a pattern that may use '
                        'chord tones -- the rule fixes harmony, not the notes.')
    p.add_argument('--bar_pattern', type=str, default=None,
                   choices=BAR_PATTERNS + [MIXED_PATTERN],
                   help='Fix the bar realisation for every song. Default: one '
                        'pattern drawn per SONG, so it is consistent within a '
                        'training window but varies across the dataset. Only '
                        'used with --cycle_unit bar.')
    p.add_argument('--no_midi', action='store_true',
                   help='Skip writing the per-song .mid files. Training never '
                        'reads them, and they are usually the bulk of the '
                        'dataset on disk.')
    p.add_argument('--keys',          type=int,   nargs='*', default=None,
                   help='Allowed root semitones (0-11). Default: all 12.')
    p.add_argument('--octave_prob',   type=float, default=0.0,
                   help='v3: chance each note is displaced an octave. The '
                        'downbeat may move register -- pitch class is '
                        'unchanged, so the rule metric is unaffected.')
    p.add_argument('--nct_prob',      type=float, default=0.0,
                   help='v3: chance a NON-downbeat note becomes a non-chord '
                        'scale tone. Requires --cycle_unit bar; the downbeat '
                        'always keeps the root.')
    p.add_argument('--seed',          type=int,   default=42)
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
        write_midi    = not args.no_midi,
        cycle_unit    = args.cycle_unit,
        pattern       = args.bar_pattern,
        allowed_keys  = args.keys,
        octave_prob   = args.octave_prob,
        nct_prob      = args.nct_prob,
        seed          = args.seed,
    )


if __name__ == '__main__':
    main()
