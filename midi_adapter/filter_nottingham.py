"""
Filter Nottingham (or any ABC/MIDI corpus) for I-IV-V-I compliant windows
and export them as CP tensor training data.

For each file the script:
  1. Extracts bar-level chord roots (via music21 for ABC, or harmonic analysis
     for MIDI using pretty_midi).
  2. Finds every 4-bar window whose root sequence matches I-IV-V-I (or any of
     its three cyclic rotations) in any key.
  3. Extracts the raw MIDI bars and converts them to CP tensor format.
  4. Saves the results in the same format as generate_synthetic_bass.py so the
     same adapter training pipeline can be used directly.

I-IV-V-I and its cyclic rotations (consecutive root diffs mod 12):
  Phase 0  I  → IV → V  → I :  diffs [5, 2, 5]
  Phase 1  IV → V  → I  → I :  diffs [2, 5, 0]
  Phase 2  V  → I  → I  → IV:  diffs [5, 0, 5]
  Phase 3  I  → I  → IV → V :  diffs [0, 5, 2]

Usage
-----
  # Nottingham ABC files
  python -m midi_adapter.filter_nottingham \\
      --input_dir data/nottingham/abc \\
      --out_dir   data/nottingham_ivvi_midi \\
      --out_pt    data/nottingham_ivvi_cp4 \\
      --require_maj

  # MIDI files (any corpus)
  python -m midi_adapter.filter_nottingham \\
      --input_dir data/nottingham/midi \\
      --out_dir   data/nottingham_ivvi_midi \\
      --out_pt    data/nottingham_ivvi_cp4
"""

from __future__ import annotations

import argparse
import os
import sys
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.generate_synthetic_bass import (
    _preprocess_pm, pitch_sort_cp,
    SUBBEATS_PER_BAR, BEATS_PER_BAR, SECONDS_PER_SUBBEAT,
    DURATION_BOUNDARIES,
)
from midi_adapter.chord_tokenizer import _NOTE_TO_SEMI, CHORD_MAP, _QUALITY_TO_IDX

try:
    import pretty_midi
except ImportError:
    print('pretty_midi is required: pip install pretty_midi')
    sys.exit(1)

# ---------------------------------------------------------------------------
# I-IV-V-I pattern  (matches OFFSETS = [0,5,7,0] in generate_synthetic_bass)
# ---------------------------------------------------------------------------

# Consecutive root-diff signatures for all 4 cyclic rotations
_IVVI_ROTATIONS = [
    (0, (5, 2, 5)),   # phase 0: I  IV  V  I
    (1, (2, 5, 0)),   # phase 1: IV  V  I  I
    (2, (5, 0, 5)),   # phase 2: V   I  I  IV
    (3, (0, 5, 2)),   # phase 3: I   I  IV  V
]

# Roots relative to the FIRST bar of the window, for each phase
# (used to reconstruct the "key" = root of the I chord)
_PHASE_TO_KEY_OFFSET = {
    0: 0,    # window starts on I  → key = window_root
    1: 7,    # window starts on IV → key = window_root - 5 = window_root + 7 (mod 12)
    2: 5,    # window starts on V  → key = window_root - 7 = window_root + 5 (mod 12)
    3: 0,    # window starts on I  → key = window_root
}


def _consecutive_diffs(roots: list[int]) -> tuple[int, ...]:
    return tuple((roots[i + 1] - roots[i]) % 12 for i in range(len(roots) - 1))


def find_ivvi_windows(
    bar_roots: list[int],      # root semitone (0-11) per bar, -1 = unknown
    require_quality: str | None = None,
    bar_qualities: list[str | None] | None = None,
) -> list[tuple[int, int, int]]:
    """Find all 4-bar windows matching I-IV-V-I (any cyclic rotation, any key).

    Returns list of (start_bar, phase, key_semitone).
    phase: 0=I-IV-V-I, 1=IV-V-I-I, 2=V-I-I-IV, 3=I-I-IV-V
    key_semitone: root of the I chord (0-11)
    """
    results = []
    n = len(bar_roots)
    for i in range(n - 3):
        roots = bar_roots[i:i + 4]
        if any(r < 0 for r in roots):
            continue
        diffs = _consecutive_diffs(roots)
        for phase, sig in _IVVI_ROTATIONS:
            if diffs == sig:
                if require_quality is not None and bar_qualities is not None:
                    quals = bar_qualities[i:i + 4]
                    if any(q != require_quality for q in quals):
                        continue
                key = (roots[0] - _PHASE_TO_KEY_OFFSET[phase]) % 12
                results.append((i, phase, key))
    return results


# ---------------------------------------------------------------------------
# Chord extraction from ABC (requires music21)
# ---------------------------------------------------------------------------

def _extract_chords_abc(
    abc_path: str,
) -> tuple[list[int], list[str | None], pretty_midi.PrettyMIDI | None]:
    """Parse an ABC file via music21 and return (bar_roots, bar_qualities, pm).

    Returns bar_roots[b] = root semitone (or -1 if no chord in bar b).
    bar_qualities[b] = quality string or None.
    pm = PrettyMIDI representation (for CP tensor extraction).
    """
    try:
        import music21 as m21
    except ImportError:
        raise ImportError("music21 is required for ABC parsing: pip install music21")

    score  = m21.converter.parse(abc_path)
    parts  = score.parts
    n_bars = max(len(p.getElementsByClass('Measure')) for p in parts)

    bar_roots:     list[int]       = [-1] * n_bars
    bar_qualities: list[str | None] = [None] * n_bars

    for part in parts:
        for measure in part.getElementsByClass('Measure'):
            b = measure.measureNumber - 1
            if b < 0 or b >= n_bars:
                continue
            for elem in measure.flatten().getElementsByClass('Harmony'):
                root_name = elem.root().name.replace('-', 'b')
                semi = _NOTE_TO_SEMI.get(root_name, -1)
                if semi < 0:
                    continue
                # Map music21 quality to our CHORD_MAP
                q_str = str(elem.chordKind) if hasattr(elem, 'chordKind') else 'major'
                quality = 'maj' if 'major' in q_str.lower() else \
                          'min' if 'minor' in q_str.lower() else \
                          '7'   if 'dominant' in q_str.lower() else 'maj'
                bar_roots[b]     = semi
                bar_qualities[b] = quality
                break   # first chord in bar wins

    # Convert score to MIDI via music21 → pretty_midi
    try:
        import io, tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tf:
            tmp = tf.name
        score.write('midi', fp=tmp)
        pm = pretty_midi.PrettyMIDI(tmp)
        os.unlink(tmp)
    except Exception:
        pm = None

    return bar_roots, bar_qualities, pm


# ---------------------------------------------------------------------------
# Chord extraction from MIDI (harmonic analysis via pretty_midi)
# ---------------------------------------------------------------------------

_CHROMA_WEIGHTS = np.array([
    2, 0.5, 1, 0.5, 1, 2, 0.5, 2, 0.5, 1, 0.5, 1   # weight root/3rd/5th more
], dtype=np.float32)


def _best_root(chroma: np.ndarray) -> int:
    """Simple chromagram-to-root: find the root of the best-fitting major triad."""
    best_root, best_score = 0, -1.0
    for r in range(12):
        score = chroma[r] + chroma[(r + 4) % 12] + chroma[(r + 7) % 12]
        if score > best_score:
            best_score = score
            best_root = r
    return best_root


def _extract_chords_midi(
    midi_path: str,
    beats_per_bar: int = BEATS_PER_BAR,
) -> tuple[list[int], list[str | None], pretty_midi.PrettyMIDI]:
    """Analyze a MIDI file harmonically to get bar-level chord roots.

    Uses chromagram-based major-triad matching.
    Returns (bar_roots, bar_qualities, pm).
    """
    pm         = pretty_midi.PrettyMIDI(midi_path)
    beat_times = pm.get_beats()
    n_beats    = len(beat_times)
    n_bars     = n_beats // beats_per_bar

    # Build per-beat chromagram
    beat_chroma = np.zeros((n_beats, 12), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            s = int(np.searchsorted(beat_times, note.start, side='right')) - 1
            e = int(np.searchsorted(beat_times, note.end,   side='right')) - 1
            s = max(0, min(s, n_beats - 1))
            e = max(s, min(e, n_beats - 1))
            pc = note.pitch % 12
            for b in range(s, e + 1):
                beat_chroma[b, pc] += 1.0

    bar_roots:     list[int]       = []
    bar_qualities: list[str | None] = []
    for b in range(n_bars):
        chroma = beat_chroma[b * beats_per_bar:(b + 1) * beats_per_bar].sum(axis=0)
        if chroma.sum() < 1e-3:
            bar_roots.append(-1)
            bar_qualities.append(None)
        else:
            root = _best_root(chroma / (chroma.sum() + 1e-8))
            bar_roots.append(root)
            bar_qualities.append('maj')   # assume major (harmonic analysis can't reliably detect quality)

    return bar_roots, bar_qualities, pm


# ---------------------------------------------------------------------------
# CP tensor extraction for a window of bars
# ---------------------------------------------------------------------------

def _extract_window_cp(
    pm: pretty_midi.PrettyMIDI,
    start_bar: int,
    n_bars: int = 4,
    max_polyphony: int = 4,
    beats_per_bar: int = BEATS_PER_BAR,
) -> torch.Tensor | None:
    """Extract CP tensor for a window of bars from a PrettyMIDI object.

    Returns (n_beats, max_polyphony * 4) uint8 tensor, or None if window is empty.
    """
    beat_times = pm.get_beats()
    start_beat = start_bar * beats_per_bar
    end_beat   = start_beat + n_bars * beats_per_bar
    if end_beat > len(beat_times):
        return None

    n_subbeats = n_bars * beats_per_bar
    rolls      = np.full((n_subbeats, max_polyphony, 4), 255, dtype=np.uint8)
    counts     = np.zeros(n_subbeats, dtype=np.uint8)

    t_start = beat_times[start_beat]
    t_end   = beat_times[end_beat] if end_beat < len(beat_times) else pm.get_end_time()

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.start < t_start or note.start >= t_end:
                continue
            abs_beat = int(np.searchsorted(beat_times, note.start, side='right')) - 1
            local_b  = abs_beat - start_beat
            if local_b < 0 or local_b >= n_subbeats:
                continue
            if counts[local_b] >= max_polyphony:
                continue
            abs_e   = int(np.searchsorted(beat_times, note.end, side='right')) - 1
            dur_idx = int(np.searchsorted(DURATION_BOUNDARIES, max(0, abs_e - abs_beat)))
            rolls[local_b, counts[local_b]] = [
                inst.program, note.pitch, dur_idx, note.velocity
            ]
            counts[local_b] += 1

    for i in range(n_subbeats):
        if counts[i] < max_polyphony:
            rolls[i, counts[i], 0] = 254   # EOS

    if counts.sum() == 0:
        return None

    data = torch.tensor(rolls.reshape(n_subbeats, max_polyphony * 4), dtype=torch.uint8)
    return pitch_sort_cp(data)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    input_dir:    str,
    out_dir:      str,
    out_pt:       str,
    require_maj:  bool = True,
    n_bars:       int  = 4,
    max_polyphony:int  = 4,
    min_notes_per_bar: int = 2,
    seed:         int  = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    all_data:   list[torch.Tensor] = []
    all_chords: list[torch.Tensor] = []
    txt_lines:  list[str]          = []
    n_songs = 0
    n_windows = 0

    exts = {'.abc', '.mid', '.midi', '.MID', '.MIDI'}
    files = sorted([
        os.path.join(root, f)
        for root, _, fnames in os.walk(input_dir)
        for f in fnames
        if os.path.splitext(f)[1] in exts
    ])
    print(f'Found {len(files)} files in {input_dir}')

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.abc':
                bar_roots, bar_qualities, pm = _extract_chords_abc(file_path)
            else:
                bar_roots, bar_qualities, pm = _extract_chords_midi(file_path)
        except Exception as e:
            print(f'  SKIP {os.path.basename(file_path)}: {e}')
            continue

        if pm is None:
            continue

        quality_filter = 'maj' if require_maj else None
        windows = find_ivvi_windows(bar_roots, require_quality=quality_filter,
                                    bar_qualities=bar_qualities)
        if not windows:
            continue

        for start_bar, phase, key in windows:
            data = _extract_window_cp(pm, start_bar, n_bars=n_bars,
                                      max_polyphony=max_polyphony)
            if data is None:
                continue
            # Check minimum note density
            n_notes = (data[:, 0::4] != 255).sum().item()
            if n_notes < n_bars * min_notes_per_bar:
                continue

            base = os.path.splitext(os.path.basename(file_path))[0]
            rel  = f'nottingham_{n_songs:06d}_key{key}_phase{phase}.mid'
            out_midi = os.path.join(out_dir, rel)

            # Save snippet as MIDI for inspection
            try:
                snippet_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
                beat_times = pm.get_beats()
                start_beat = start_bar * BEATS_PER_BAR
                end_beat   = start_beat + n_bars * BEATS_PER_BAR
                t0 = beat_times[start_beat]
                t1 = beat_times[end_beat] if end_beat < len(beat_times) else pm.get_end_time()
                imap = {}
                for inst in pm.instruments:
                    if inst.is_drum:
                        continue
                    for note in inst.notes:
                        if note.start < t0 or note.start >= t1:
                            continue
                        if inst.program not in imap:
                            imap[inst.program] = pretty_midi.Instrument(program=inst.program)
                            snippet_pm.instruments.append(imap[inst.program])
                        imap[inst.program].notes.append(pretty_midi.Note(
                            velocity=note.velocity, pitch=note.pitch,
                            start=note.start - t0, end=note.end - t0,
                        ))
                snippet_pm.write(out_midi)
            except Exception:
                pass

            all_data.append(data)
            # Store chord tokens (bar-level key encoding, one token per beat)
            from midi_adapter.chord_tokenizer import chord_str_to_token
            ROOT_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            from midi_adapter.generate_synthetic_bass import OFFSETS
            beat_chord_tokens = []
            for beat_in_window in range(n_bars * BEATS_PER_BAR):
                bar_in_window = beat_in_window // BEATS_PER_BAR
                root = (key + OFFSETS[(bar_in_window + phase) % 4]) % 12
                beat_chord_tokens.append(chord_str_to_token(f'{ROOT_NAMES[root]}:maj'))
            all_chords.append(torch.tensor(beat_chord_tokens, dtype=torch.int16))
            txt_lines.append(f'{n_songs}\t{rel}')
            n_songs += 1

        n_windows += len(windows)
        if n_songs % 100 == 0 and n_songs > 0:
            print(f'  {n_songs} windows exported ({os.path.basename(file_path)})')

    if n_songs == 0:
        print('No matching windows found. Try --no_require_maj or check your input_dir.')
        return

    n_subbeats = n_bars * BEATS_PER_BAR
    torch.save(torch.cat(all_data, dim=0),                f'{out_pt}.pt')
    torch.save(torch.tensor([n_subbeats] * n_songs),      f'{out_pt}.length.pt')
    torch.save(torch.zeros(n_songs, 2, dtype=torch.int8), f'{out_pt}.pitch_shift_range.pt')
    torch.save(all_chords,                                 f'{out_pt}.beat_chords.pt')
    with open(f'{out_pt}.txt', 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    print(f'\nDone. {n_songs} windows from {len(files)} files.')
    print(f'Saved → {out_pt}.{{pt,length.pt,pitch_shift_range.pt,beat_chords.pt,txt}}')
    print(f'MIDI snippets → {out_dir}/')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Filter corpus for I-IV-V-I windows')
    p.add_argument('--input_dir',   required=True,
                   help='Directory of .abc or .mid files (searched recursively)')
    p.add_argument('--out_dir',     required=True,
                   help='Directory for extracted MIDI snippets')
    p.add_argument('--out_pt',      required=True,
                   help='Output prefix for CP tensor dataset files')
    p.add_argument('--require_maj', action='store_true',
                   help='Only accept windows where all 4 chords are major quality '
                        '(stricter but consistent with CPChordRuleModel)')
    p.add_argument('--no_require_maj', dest='require_maj', action='store_false',
                   help='Accept windows regardless of chord quality (more data)')
    p.set_defaults(require_maj=True)
    p.add_argument('--n_bars',      type=int, default=4,
                   help='Number of bars per window (default 4 = one full cycle)')
    p.add_argument('--max_polyphony', type=int, default=4)
    p.add_argument('--min_notes_per_bar', type=int, default=2,
                   help='Discard windows with fewer than this many notes per bar')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    build_dataset(
        input_dir     = args.input_dir,
        out_dir       = args.out_dir,
        out_pt        = args.out_pt,
        require_maj   = args.require_maj,
        n_bars        = args.n_bars,
        max_polyphony = args.max_polyphony,
        min_notes_per_bar = args.min_notes_per_bar,
        seed          = args.seed,
    )


if __name__ == '__main__':
    main()
