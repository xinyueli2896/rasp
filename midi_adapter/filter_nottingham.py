"""
Two-step pipeline for extracting I-IV-V-I windows from a MIDI corpus.

Uses a downbeat-aligned loader (`_load_midi_aligned`) that:
  * starts bar 0 at the first musical downbeat (handles pickup beats)
  * builds a per-MIDI subbeat grid by interpolating BEAT_DIV positions inside
    each pm.get_beats() interval (handles tempo changes)
  * quantizes every note onset/offset to that grid

This replaces the earlier `preprocess_large_midi_dataset.preprocess_midi`
path, which used UglyMIDI/xf_midi with constant_tempo rescaling but did NOT
align to musical downbeats — pickups bled into "bar 0" and shifted every
4-bar window's chord detection.

Step 1 — filter (loads MIDIs, finds I-IV-V-I windows):
  python -m midi_adapter.filter_nottingham filter \\
      --input_dir data/nottingham/MIDI \\
      --manifest  data/nottingham_ivvi_manifest.json

Step 2 — extract (slices cached CP tensors per window, optionally transposes
to a target-key set):
  python -m midi_adapter.filter_nottingham extract \\
      --manifest  data/nottingham_ivvi_manifest.json \\
      --out_dir   data/nottingham_ivvi_midi \\
      --out_pt    data/nottingham_ivvi_cp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pretty_midi
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.generate_synthetic_bass import (
    pitch_sort_cp,
    BEAT_DIV, SUBBEATS_PER_BAR, BEATS_PER_BAR, OFFSETS,
    DURATION_TEMPLATES, DURATION_BOUNDARIES,
)

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Default window length in bars. At chords_per_bar=2, an 8-bar window has 16
# chord positions = 4 full I-IV-V-I cycles — plenty of rule signal for the
# adapter while still finding lots of matches in folk corpora.
N_BARS_WINDOW = 8

# Default chord rate. 2 chords per bar (= half-bar harmonic rhythm) is more
# musically natural for 4/4 folk/pop than 1 chord per bar, and matches what
# the chord-detection step already produces at half-bar granularity.
CHORDS_PER_BAR_DEFAULT = 2


# ---------------------------------------------------------------------------
# Downbeat-aligned loader.
#
# Why not preprocess_large_midi_dataset.preprocess_midi?
#   That uses UglyMIDI(constant_tempo=...) which rescales the time axis but
#   does NOT align to the first musical downbeat. If a Nottingham tune has a
#   pickup, "bar 0" in the resulting CP tensor straddles the pickup and the
#   start of musical bar 1 — chord detection then mixes pickup notes into the
#   I chord and the 4-bar windows are off by some number of subbeats.
#
# This loader:
#   1. Uses pm.get_downbeats() to find the first musical downbeat
#   2. Uses pm.get_beats() with linear interpolation inside each beat to build
#      a per-MIDI subbeat grid that respects the file's actual tempo changes
#   3. Quantizes every note onset/offset to that grid and writes a CP tensor
# ---------------------------------------------------------------------------

def _load_midi_aligned(midi_path: str, max_polyphony: int = 4,
                        align_to_downbeat: bool = True) -> np.ndarray | None:
    """Load a MIDI, quantize to 16th-note subbeats (using the file's own tempo
    curve), and return a CP tensor.

    align_to_downbeat=True (default, for Nottingham/folk):
        Bar 0 starts at pm.get_downbeats()[0], so any pickup / anacrusis is
        skipped. Files not in pure 4/4 are rejected.

    align_to_downbeat=False (for datasets already well-adjusted, e.g. POP909):
        Bar 0 starts at pm.get_beats()[0]. No downbeat search, no time-sig
        rejection — trust the file's timing.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    if align_to_downbeat:
        # Reject anything that isn't pure 4/4. An empty time_signature_changes
        # list defaults to 4/4 in MIDI semantics, so we accept that.
        for ts in pm.time_signature_changes:
            if ts.numerator != 4 or ts.denominator != 4:
                return None

    beats = pm.get_beats()
    if len(beats) < 2:
        return None

    if align_to_downbeat:
        downbeats = pm.get_downbeats()
        if len(downbeats) == 0:
            return None
        t_start_db = downbeats[0]
        start_beat_idx = int(np.searchsorted(beats, t_start_db - 1e-9, side='left'))
    else:
        start_beat_idx = 0
    if start_beat_idx >= len(beats) - 1:
        return None
    t_start_db = beats[start_beat_idx]

    # Build subbeat times by interpolating BEAT_DIV positions inside each beat.
    subbeat_times: list[float] = []
    for i in range(start_beat_idx, len(beats) - 1):
        dt = (beats[i + 1] - beats[i]) / BEAT_DIV
        for j in range(BEAT_DIV):
            subbeat_times.append(beats[i] + j * dt)
    subbeat_times.append(float(beats[-1]))
    subbeat_times_arr = np.asarray(subbeat_times)
    n_subbeats = len(subbeat_times_arr)
    if n_subbeats < SUBBEATS_PER_BAR:
        return None

    rolls  = np.full((n_subbeats, max_polyphony, 4), 255, dtype=np.uint8)
    counts = np.zeros(n_subbeats, dtype=np.uint8)

    t_end = subbeat_times_arr[-1]
    for inst in pm.instruments:
        prog = 127 if inst.is_drum else inst.program
        for note in inst.notes:
            if note.start < t_start_db or note.start >= t_end:
                continue
            s = int(np.searchsorted(subbeat_times_arr, note.start, side='right')) - 1
            e = int(np.searchsorted(subbeat_times_arr, note.end,   side='right')) - 1
            if s < 0 or s >= n_subbeats or counts[s] >= max_polyphony:
                continue
            dur_idx = int(np.searchsorted(DURATION_BOUNDARIES, max(0, e - s)))
            rolls[s, counts[s]] = [prog, note.pitch, dur_idx, note.velocity]
            counts[s] += 1

    for i in range(n_subbeats):
        if counts[i] < max_polyphony:
            rolls[i, counts[i], 0] = 254   # EOS

    data = torch.tensor(rolls.reshape(n_subbeats, max_polyphony * 4), dtype=torch.uint8)
    return pitch_sort_cp(data).numpy()


# ---------------------------------------------------------------------------
# Chord detection from CP tensor
# ---------------------------------------------------------------------------

def _chromagram_to_root(chroma: np.ndarray) -> int:
    """Return the best major-triad root for a 12-bin chromagram, or -1 if empty."""
    if chroma.sum() < 1e-3:
        return -1
    c = chroma / chroma.sum()
    best_r, best_s = 0, -1.0
    for r in range(12):
        s = c[r] + c[(r + 4) % 12] + c[(r + 7) % 12]
        if s > best_s:
            best_s, best_r = s, r
    return best_r


def _extract_chord_roots_from_cp(cp_data: np.ndarray,
                                  chords_per_bar: int = 2) -> list[int]:
    """Detect chord root per (1/chords_per_bar) bar via chromagram matching.

    chords_per_bar=1 → one root per bar  (16 subbeats per chord at beat_div=4)
    chords_per_bar=2 → one root per HALF-bar (8 subbeats per chord) [default]

    Returns a list of length n_bars * chords_per_bar. Empty slices return -1.
    """
    assert SUBBEATS_PER_BAR % chords_per_bar == 0, \
        f'SUBBEATS_PER_BAR={SUBBEATS_PER_BAR} not divisible by chords_per_bar={chords_per_bar}'
    sub_per_chord = SUBBEATS_PER_BAR // chords_per_bar

    n_subbeats = cp_data.shape[0]
    n_voices   = cp_data.shape[1] // 4
    n_chords   = n_subbeats // sub_per_chord

    chroma = np.zeros((n_chords, 12), dtype=np.float32)
    progs   = cp_data[:, 0::4]
    pitches = cp_data[:, 1::4]
    for sb in range(n_chords * sub_per_chord):
        c = sb // sub_per_chord
        for v in range(n_voices):
            prog  = int(progs[sb, v])
            pitch = int(pitches[sb, v])
            if prog == 255 or prog == 254 or prog == 127:
                continue
            if pitch == 255:
                continue
            chroma[c, pitch % 12] += 1.0

    return [_chromagram_to_root(chroma[c]) for c in range(n_chords)]


# Back-compat shim — old name kept for any callers.
def _extract_bar_roots_from_cp(cp_data: np.ndarray, chords_per_bar: int = 2) -> list[int]:
    return _extract_chord_roots_from_cp(cp_data, chords_per_bar=chords_per_bar)


def _find_ivvi_windows(chord_roots: list[int], n_bars: int = 4,
                       chords_per_bar: int = 2,
                       phase_zero_only: bool = True) -> list[dict]:
    """Return list of {start_bar, phase, key} dicts whose detected chord-root
    sequence matches the I-IV-V-I rule at chords_per_bar harmonic rhythm:

        chord_roots[i + j] == (key + OFFSETS[(j + phase) % 4]) % 12  for all j

    A window covers `n_bars * chords_per_bar` chord positions, which must be a
    multiple of 4 (= integer number of I-IV-V-I cycles). i steps in chord
    positions; start_bar is reported as i // chords_per_bar (whole bars only)
    so we never cut mid-bar.
    """
    n_chords_window = n_bars * chords_per_bar
    assert n_chords_window % 4 == 0, \
        f'n_bars * chords_per_bar = {n_chords_window} must be a multiple of 4'
    phases = (0,) if phase_zero_only else (0, 1, 2, 3)
    results = []
    # Only start on bar boundaries (multiples of chords_per_bar) so windows
    # always align to musical bar starts.
    for i in range(0, len(chord_roots) - n_chords_window + 1, chords_per_bar):
        roots = chord_roots[i:i + n_chords_window]
        if any(r < 0 for r in roots):
            continue
        matched = False
        for phase in phases:
            for key in range(12):
                if all(roots[j] == (key + OFFSETS[(j + phase) % 4]) % 12
                       for j in range(n_chords_window)):
                    results.append({
                        'start_bar': i // chords_per_bar,
                        'phase':     phase,
                        'key':       key,
                    })
                    matched = True
                    break
            if matched:
                break
    return results


# ---------------------------------------------------------------------------
# MIDI snippet writer — render directly from the quantized CP tensor at a
# normal 120 BPM. This matches what the model actually trains on (notes snap
# to 16th-note subbeats; one bar = 2 seconds = 16 subbeats * 0.125 s), instead
# of the original variable-tempo timing that may have pickups / wobble.
# ---------------------------------------------------------------------------

def _save_midi_example(cp_data: np.ndarray, start_bar: int, n_bars: int,
                      max_polyphony: int, out_path: str,
                      tempo: float = 120.0) -> bool:
    """Render bars [start_bar, start_bar + n_bars) of `cp_data` as a quantized
    MIDI snippet at the given tempo. Returns True iff at least one note was
    written."""
    start_sb   = start_bar * SUBBEATS_PER_BAR
    n_subbeats = n_bars * SUBBEATS_PER_BAR
    if start_sb + n_subbeats > cp_data.shape[0]:
        return False
    seconds_per_subbeat = 60.0 / tempo / BEAT_DIV

    out_pm    = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst_map: dict[int, pretty_midi.Instrument] = {}
    any_note  = False

    for local_sb in range(n_subbeats):
        sb = start_sb + local_sb
        start = local_sb * seconds_per_subbeat
        for v in range(max_polyphony):
            prog    = int(cp_data[sb, v * 4 + 0])
            pitch   = int(cp_data[sb, v * 4 + 1])
            dur_idx = int(cp_data[sb, v * 4 + 2])
            vel     = int(cp_data[sb, v * 4 + 3])
            if prog == 254 or prog == 255:   # EOS / pad: no further voices this subbeat
                break
            if pitch == 255 or dur_idx >= len(DURATION_TEMPLATES):
                continue
            end = (local_sb + int(DURATION_TEMPLATES[dur_idx])) * seconds_per_subbeat
            is_drum = (prog == 127)
            key = prog
            if key not in inst_map:
                inst_map[key] = pretty_midi.Instrument(
                    program=0 if is_drum else prog, is_drum=is_drum)
                out_pm.instruments.append(inst_map[key])
            inst_map[key].notes.append(pretty_midi.Note(
                velocity=vel if vel > 0 else 80, pitch=pitch, start=start, end=end))
            any_note = True

    if not any_note:
        return False
    out_pm.write(out_path)
    return True


# ---------------------------------------------------------------------------
# Snippet writer — preserves the source MIDI's track structure. Every
# instrument keeps its program, name, drum flag; only the notes get sliced to
# bars [start_bar, start_bar + n_bars) and quantized onto the same downbeat-
# aligned 16th-note grid used by _load_midi_aligned. Rendered at 120 BPM.
# ---------------------------------------------------------------------------

def _save_original_tracks_snippet(midi_path: str, start_bar: int, n_bars: int,
                                   out_path: str,
                                   align_to_downbeat: bool = True,
                                   quantize: bool = True,
                                   tempo: float = 120.0) -> bool:
    """Slice bars [start_bar, start_bar + n_bars) from the source MIDI.

    align_to_downbeat: anchor bar 0 to pm.get_downbeats()[0] (default True).
    quantize:
        True  → snap every note's start/end to the subbeat grid built by
                interpolating pm.get_beats(); render at fixed 120 BPM.
        False → pure cut: original note times are preserved (only shifted by
                -t_win_start) and the source's tempo is kept so the snippet
                sounds identical to the original in that time range.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    beats = pm.get_beats()
    if len(beats) < 2:
        return False

    if align_to_downbeat:
        downbeats = pm.get_downbeats()
        if len(downbeats) == 0:
            return False
        start_beat_idx = int(np.searchsorted(beats, downbeats[0] - 1e-9, side='left'))
    else:
        start_beat_idx = 0
    if start_beat_idx >= len(beats) - 1:
        return False

    # Compute the window boundaries in original-tempo seconds.
    start_beat = start_bar * BEATS_PER_BAR
    end_beat   = start_beat + n_bars * BEATS_PER_BAR
    idx_start  = start_beat_idx + start_beat
    idx_end    = start_beat_idx + end_beat
    if idx_end > len(beats) - 1:
        return False
    t_win_start = float(beats[idx_start])
    if idx_end < len(beats):
        t_win_end = float(beats[idx_end])
    else:
        # Extrapolate one beat past the last beat using its local tempo.
        t_win_end = float(beats[-1] + (beats[-1] - beats[-2]))

    eps = 1e-4

    if not quantize:
        # Pure cut: preserve original note timings + tempo, no grid snapping.
        tempos, _ = pm.get_tempo_changes()
        src_tempo = float(tempos[0]) if len(tempos) else 120.0
        out_pm    = pretty_midi.PrettyMIDI(initial_tempo=src_tempo)
        any_note  = False
        for inst in pm.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for note in inst.notes:
                if note.end <= t_win_start + eps or note.start >= t_win_end - eps:
                    continue
                new_inst.notes.append(pretty_midi.Note(
                    velocity = note.velocity,
                    pitch    = note.pitch,
                    start    = max(0.0, note.start - t_win_start),
                    end      = max(1e-4, min(note.end, t_win_end) - t_win_start),
                ))
            if new_inst.notes:
                out_pm.instruments.append(new_inst)
                any_note = True
        if not any_note:
            return False
        out_pm.write(out_path)
        return True

    # Quantized path: snap onto the interpolated 16th-note grid, render at 120.
    subbeat_times: list[float] = []
    for i in range(start_beat_idx, len(beats) - 1):
        dt = (beats[i + 1] - beats[i]) / BEAT_DIV
        for j in range(BEAT_DIV):
            subbeat_times.append(beats[i] + j * dt)
    subbeat_times.append(float(beats[-1]))
    subbeat_times_arr = np.asarray(subbeat_times)

    start_sb = start_bar * SUBBEATS_PER_BAR
    end_sb   = start_sb + n_bars * SUBBEATS_PER_BAR
    if end_sb > len(subbeat_times_arr):
        return False

    out_pm     = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    sec_per_sb = 60.0 / tempo / BEAT_DIV
    any_note   = False

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            if note.end <= t_win_start + eps or note.start >= t_win_end - eps:
                continue
            clip_start = max(note.start, t_win_start)
            clip_end   = min(note.end,   t_win_end)
            s_abs = int(np.searchsorted(subbeat_times_arr, clip_start + eps, side='right')) - 1
            e_abs = int(np.searchsorted(subbeat_times_arr, clip_end   + eps, side='right')) - 1
            s_loc = max(0, s_abs - start_sb)
            e_loc = max(s_loc + 1, min(e_abs - start_sb, n_bars * SUBBEATS_PER_BAR))
            if s_loc >= n_bars * SUBBEATS_PER_BAR:
                continue
            new_inst.notes.append(pretty_midi.Note(
                velocity = note.velocity,
                pitch    = note.pitch,
                start    = s_loc * sec_per_sb,
                end      = e_loc * sec_per_sb,
            ))
        if new_inst.notes:
            out_pm.instruments.append(new_inst)
            any_note = True

    if not any_note:
        return False
    out_pm.write(out_path)
    return True


# ---------------------------------------------------------------------------
# Polyphony check — Structured-Arrangement needs both melody and chord. A
# window with no polyphony (all subbeats have ≤1 note) can't be split into
# melody+chord and is rejected by the filter.
# ---------------------------------------------------------------------------

def _count_polyphonic_subbeats(cp_data: np.ndarray, start_sb: int,
                                n_subbeats: int, max_polyphony: int = 4) -> int:
    """Count subbeats with ≥2 simultaneous non-drum, non-pad notes."""
    progs   = cp_data[start_sb:start_sb + n_subbeats, 0::4]
    pitches = cp_data[start_sb:start_sb + n_subbeats, 1::4]
    # A slot holds a usable note iff program < 127 (not drum/EOS/pad) and pitch != 255
    real = (progs < 127) & (pitches != 255)
    per_sb_counts = real.sum(axis=1)   # shape (n_subbeats,)
    return int((per_sb_counts >= 2).sum())


# ---------------------------------------------------------------------------
# Step 1: Filter
# ---------------------------------------------------------------------------

def cmd_filter(args):
    exts  = {'.mid', '.midi', '.MID', '.MIDI'}
    files = sorted([
        os.path.join(root, f)
        for root, _, fnames in os.walk(args.input_dir)
        for f in fnames if os.path.splitext(f)[1] in exts
    ])
    print(f'Scanning {len(files)} MIDI files in {args.input_dir} ...')

    manifest  = []
    n_windows = 0
    n_skipped = 0

    from collections import Counter
    examples_per_key = Counter()
    if args.save_examples_dir:
        # Wipe any examples saved by a previous run so the directory only
        # contains snippets matching the current filter settings.
        import shutil
        if os.path.isdir(args.save_examples_dir):
            shutil.rmtree(args.save_examples_dir)
        os.makedirs(args.save_examples_dir, exist_ok=True)

    for midi_path in files:
        try:
            cp_data = _load_midi_aligned(
                midi_path,
                max_polyphony     = args.max_polyphony,
                align_to_downbeat = not args.no_align,
            )
        except Exception as e:
            n_skipped += 1
            print(f'  SKIP {os.path.basename(midi_path)}: {e}')
            continue

        if cp_data is None:
            n_skipped += 1
            continue

        chord_roots = _extract_chord_roots_from_cp(
            cp_data, chords_per_bar=args.chords_per_bar)
        windows = _find_ivvi_windows(
            chord_roots, n_bars=args.n_bars,
            chords_per_bar=args.chords_per_bar,
            phase_zero_only=not args.all_phases)
        n_subbeats_window = args.n_bars * SUBBEATS_PER_BAR
        for w in windows:
            start_sb = w['start_bar'] * SUBBEATS_PER_BAR
            n_poly   = _count_polyphonic_subbeats(
                cp_data, start_sb, n_subbeats_window, args.max_polyphony)
            if n_poly < args.min_polyphonic_subbeats:
                continue   # no chord content — can't form a melody+chord lead sheet

            manifest.append({
                'midi_path':  midi_path,
                'start_bar':  w['start_bar'],
                'phase':      w['phase'],
                'key':        w['key'],
                'key_name':   ROOT_NAMES[w['key']],
                'n_bars':     args.n_bars,
                'chord_roots': chord_roots[
                    w['start_bar'] * args.chords_per_bar:
                    (w['start_bar'] + args.n_bars) * args.chords_per_bar],
                'chords_per_bar': args.chords_per_bar,
                'n_polyphonic_subbeats': n_poly,
            })
            n_windows += 1
            if args.save_examples_dir and examples_per_key[w['key']] < args.examples_per_key:
                stem = os.path.splitext(os.path.basename(midi_path))[0]
                out  = os.path.join(
                    args.save_examples_dir,
                    f'key{ROOT_NAMES[w["key"]]}_phase{w["phase"]}_bar{w["start_bar"]:04d}_{stem}.mid',
                )
                try:
                    if _save_original_tracks_snippet(
                            midi_path, w['start_bar'], args.n_bars, out,
                            align_to_downbeat=not args.no_align):
                        examples_per_key[w['key']] += 1
                except Exception as e:
                    print(f'  example-save failed for {midi_path}: {e}')

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nFound {n_windows} windows across {len(files) - n_skipped} files '
          f'({n_skipped} skipped).')
    print(f'Manifest saved → {args.manifest}')
    if args.save_examples_dir:
        print(f'Saved {sum(examples_per_key.values())} example snippets '
              f'→ {args.save_examples_dir}')
    print('\nKey distribution:')
    from collections import Counter
    key_counts = Counter(e['key_name'] for e in manifest)
    for k, v in sorted(key_counts.items(), key=lambda x: -x[1]):
        print(f'  {k:<4} {v}')


# ---------------------------------------------------------------------------
# Step 2: Extract — slice cached CP tensors by manifest entries
# ---------------------------------------------------------------------------

# Default training key set: all 12 pitch classes except F# (6) and G# (8),
# which are reserved as unseen-eval keys.
SEEN_KEYS_DEFAULT   = (0, 1, 2, 3, 4, 5, 7, 9, 10, 11)
UNSEEN_KEYS_DEFAULT = (6, 8)


def _signed_shift(target_key: int, base_key: int) -> int:
    """Smallest-magnitude semitone shift mapping base_key → target_key.
    Returns a value in [-5, +6]."""
    s = (target_key - base_key) % 12
    return s - 12 if s > 6 else s


def _transpose_window(window: torch.Tensor, shift: int) -> torch.Tensor | None:
    """Transpose a CP window by `shift` semitones (matches cp_transformer.preprocess:
    pitch slot += pitch_shift * is_not_drum). Returns None if any non-drum note
    would land outside MIDI range [0, 127]."""
    if shift == 0:
        return window.clone()
    win     = window.clone()
    progs   = win[:, 0::4]
    pitches = win[:, 1::4]
    # Valid note: program in 0..126 (excludes drum=127, EOS=254, pad=255)
    # and pitch != 255 (pad).
    valid   = (progs < 127) & (pitches != 255)
    shifted = pitches.long() + shift
    if ((shifted < 0) | (shifted > 127))[valid].any():
        return None
    pitches_new = shifted.to(pitches.dtype)
    new = pitches.clone()
    new[valid] = pitches_new[valid]
    win[:, 1::4] = new
    return win


def cmd_extract(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f'Loaded {len(manifest)} windows from {args.manifest}')

    os.makedirs(args.out_dir, exist_ok=True)

    from midi_adapter.chord_tokenizer import chord_str_to_token

    all_data:   list[torch.Tensor] = []
    all_chords: list[torch.Tensor] = []
    txt_lines:  list[str] = []
    n_saved = 0

    # Cache preprocessed CP tensors per file to avoid re-loading
    cp_cache: dict[str, np.ndarray] = {}
    cache_order: list[str] = []

    n_bars_window     = manifest[0].get('n_bars', N_BARS_WINDOW) if manifest else N_BARS_WINDOW
    n_subbeats_window = n_bars_window * SUBBEATS_PER_BAR
    entry_chords_per_bar = manifest[0].get('chords_per_bar', CHORDS_PER_BAR_DEFAULT) \
                           if manifest else CHORDS_PER_BAR_DEFAULT
    print(f'Window length: {n_bars_window} bars ({n_subbeats_window} subbeats),  '
          f'chords_per_bar={entry_chords_per_bar}')

    for entry in manifest:
        midi_path = entry['midi_path']
        if midi_path not in cp_cache:
            try:
                cp_arr = _load_midi_aligned(
                    midi_path,
                    max_polyphony     = args.max_polyphony,
                    align_to_downbeat = not args.no_align,
                )
            except Exception as e:
                print(f'  SKIP {midi_path}: {e}')
                cp_cache[midi_path] = None
                continue
            if cp_arr is None:
                cp_cache[midi_path] = None
                continue
            cp_cache[midi_path] = cp_arr
            cache_order.append(midi_path)
            # Bound cache size
            if len(cache_order) > 200:
                old = cache_order.pop(0)
                cp_cache.pop(old, None)

        cp_data = cp_cache[midi_path]
        if cp_data is None:
            continue

        start_sb = entry['start_bar'] * SUBBEATS_PER_BAR
        end_sb   = start_sb + n_subbeats_window
        if end_sb > cp_data.shape[0]:
            continue

        base_window = torch.tensor(cp_data[start_sb:end_sb].copy(), dtype=torch.uint8)
        base_window = pitch_sort_cp(base_window)

        n_notes = (base_window[:, 0::4] < 254).sum().item()
        if n_notes < n_bars_window * args.min_notes_per_bar:
            continue

        base_key = entry['key']
        phase    = entry['phase']

        for target_key in args.target_keys:
            shift = _signed_shift(target_key, base_key)
            win_t = _transpose_window(base_window, shift)
            if win_t is None:
                continue   # transposition would clip the MIDI range
            win_t = pitch_sort_cp(win_t)

            beat_tokens = []
            sub_per_chord = SUBBEATS_PER_BAR // entry_chords_per_bar
            for sb in range(n_subbeats_window):
                chord_in_window = sb // sub_per_chord
                root = (target_key + OFFSETS[(chord_in_window + phase) % 4]) % 12
                beat_tokens.append(chord_str_to_token(f'{ROOT_NAMES[root]}:maj'))
            all_chords.append(torch.tensor(beat_tokens, dtype=torch.int16))

            rel = f'nottingham_{n_saved:06d}_key{target_key}_phase{phase}.mid'
            txt_lines.append(f'{n_saved}\t{rel}')
            all_data.append(win_t)
            n_saved += 1

        if n_saved % 200 == 0 and n_saved > 0:
            print(f'  {n_saved} windows extracted ...')

    if n_saved == 0:
        print('No windows passed the note-density filter. Try --min_notes_per_bar 1')
        return

    torch.save(torch.cat(all_data, dim=0),                f'{args.out_pt}.pt')
    torch.save(torch.tensor([n_subbeats_window] * n_saved), f'{args.out_pt}.length.pt')
    torch.save(torch.zeros(n_saved, 2, dtype=torch.int8), f'{args.out_pt}.pitch_shift_range.pt')
    torch.save(all_chords,                                 f'{args.out_pt}.beat_chords.pt')
    with open(f'{args.out_pt}.txt', 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    print(f'\nSaved {n_saved} windows across target_keys={list(args.target_keys)}.')
    print(f'Dataset → {args.out_pt}.{{pt,length.pt,pitch_shift_range.pt,beat_chords.pt,txt}}')

    from collections import Counter
    key_counts = Counter(
        int(line.split('key')[1].split('_')[0]) for line in txt_lines
    )
    print('Per-key window counts:')
    for k in sorted(key_counts):
        print(f'  {ROOT_NAMES[k]:<4} {key_counts[k]}')


# ---------------------------------------------------------------------------
# Step 3 (optional): Lead-sheet export for Structured-Arrangement
# ---------------------------------------------------------------------------

def cmd_leadsheets(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f'Loaded {len(manifest)} matched windows from {args.manifest}')
    os.makedirs(args.out_dir, exist_ok=True)

    from collections import Counter
    per_key = Counter()
    n_saved = 0

    for entry in manifest:
        if args.limit and n_saved >= args.limit:
            break
        key = entry['key']
        if args.per_key and per_key[key] >= args.per_key:
            continue

        midi_path = entry['midi_path']
        start_bar = entry['start_bar']
        phase     = entry['phase']
        n_bars    = entry.get('n_bars', N_BARS_WINDOW)

        stem = os.path.splitext(os.path.basename(midi_path))[0]
        out  = os.path.join(
            args.out_dir,
            f'{n_saved:06d}_key{ROOT_NAMES[key]}_phase{phase}_bar{start_bar:04d}_{stem}.mid',
        )
        try:
            if _save_original_tracks_snippet(
                    midi_path, start_bar, n_bars, out,
                    align_to_downbeat=not args.no_align,
                    quantize          = not args.no_quantize):
                per_key[key] += 1
                n_saved += 1
        except Exception as e:
            print(f'  SKIP {midi_path}: {e}')

    print(f'\nSaved {n_saved} lead-sheet snippets → {args.out_dir}')
    print('Per-key counts:')
    for k in sorted(per_key):
        print(f'  {ROOT_NAMES[k]:<4} {per_key[k]}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Filter MIDI corpus for I-IV-V-I windows '
                                            '(using preprocess_large_midi_dataset.preprocess_midi)')
    sub = p.add_subparsers(dest='cmd', required=True)

    pf = sub.add_parser('filter', help='Scan MIDI files and save matching-window manifest')
    pf.add_argument('--input_dir',     required=True,
                    help='Directory of MIDI files (searched recursively)')
    pf.add_argument('--manifest',      required=True)
    pf.add_argument('--max_polyphony', type=int, default=4)
    pf.add_argument('--n_bars',        type=int, default=N_BARS_WINDOW,
                    help='Window length in bars; n_bars * chords_per_bar must be a '
                         'multiple of 4. Default 8 (2 I-IV-V-I cycles at chords_per_bar=2).')
    pf.add_argument('--chords_per_bar', type=int, default=CHORDS_PER_BAR_DEFAULT,
                    choices=[1, 2, 4],
                    help='Harmonic rhythm in chords per bar. 2 (default) = 8 subbeats '
                         'per chord, gives a 2-bar I-IV-V-I cadence; 1 = 16 subbeats '
                         'per chord, gives a 4-bar cadence.')
    pf.add_argument('--all_phases',    action='store_true',
                    help='Accept all 4 cyclic rotations (start on I/IV/V/I). '
                         'Default keeps only phase 0 (windows start on I).')
    pf.add_argument('--no_align',      action='store_true',
                    help='Skip downbeat alignment and the 4/4 time-signature check. '
                         'Bar 0 starts at pm.get_beats()[0]. Use for datasets that '
                         'are already well-adjusted (e.g. POP909) where every song '
                         'starts on the downbeat and quantization is clean.')
    pf.add_argument('--min_polyphonic_subbeats', type=int, default=8,
                    help='Reject windows with fewer than this many subbeats that '
                         'have ≥2 simultaneous notes — guarantees the matched '
                         'window has enough chord content to split into melody '
                         '+ chord tracks for Structured-Arrangement input.')
    pf.add_argument('--save_examples_dir', type=str, default=None,
                    help='If set, save snippets of qualifying windows here. Track '
                         'structure is preserved exactly (every source instrument '
                         'stays on its own track with original program/name). Notes '
                         'are quantized onto the same 16th-note grid as the CP tensor '
                         'and rendered at 120 BPM. Wiped at the start of each run. '
                         '(For skyline melody/chord snippets, use the `leadsheets` '
                         'subcommand.)')
    pf.add_argument('--examples_per_key',  type=int, default=3,
                    help='Cap on saved examples per detected key (default 3).')

    pe = sub.add_parser('extract', help='Extract CP tensors from a manifest')
    pe.add_argument('--manifest',       required=True)
    pe.add_argument('--out_dir',        required=True)
    pe.add_argument('--out_pt',         required=True)
    pe.add_argument('--max_polyphony',  type=int, default=4)
    pe.add_argument('--min_notes_per_bar', type=int, default=2)
    pe.add_argument('--no_align',       action='store_true',
                    help='Skip downbeat alignment / 4/4 check (use for POP909 etc.). '
                         'Must match the --no_align used in the filter step.')
    pe.add_argument('--target_keys',    type=int, nargs='+', default=list(SEEN_KEYS_DEFAULT),
                    help='Pitch classes (0-11) to transpose every window into. Each '
                         'original window is duplicated len(target_keys) times, one per '
                         'target key (smallest signed shift). Default: all 12 keys '
                         'except F# (6) and G# (8), reserved as unseen-eval keys. '
                         'Pass "6 8" to build the unseen-eval dataset.')

    pl = sub.add_parser('leadsheets',
                        help='Save matched windows as multi-track MIDI files with '
                             'the source MIDI\'s track structure preserved exactly, '
                             'ready for Structured-Arrangement / AccoMontage input '
                             '(they accept arbitrary multi-track input).')
    pl.add_argument('--manifest',      required=True)
    pl.add_argument('--out_dir',       required=True)
    pl.add_argument('--no_align',      action='store_true',
                    help='Skip downbeat alignment / 4/4 check (use for POP909 etc.). '
                         'Must match the --no_align used in the filter step.')
    pl.add_argument('--no_quantize',   action='store_true',
                    help='Pure time-based cut: keep original note timings AND the '
                         'source tempo. When set the snippet plays back exactly like '
                         'the corresponding time range of the source file, no '
                         'snapping to a 16th-note grid.')
    pl.add_argument('--limit',         type=int, default=0,
                    help='Cap total snippets saved (0 = no cap).')
    pl.add_argument('--per_key',       type=int, default=0,
                    help='Cap snippets per detected key (0 = no cap).')

    args = p.parse_args()
    if args.cmd == 'filter':
        cmd_filter(args)
    elif args.cmd == 'extract':
        cmd_extract(args)
    else:
        cmd_leadsheets(args)


if __name__ == '__main__':
    main()
