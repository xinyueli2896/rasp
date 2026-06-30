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

# Default window length in bars; must be a multiple of 4 (= integer number of
# I-IV-V-I cycles). 8 bars (= 2 cycles, 128 subbeats) is a good balance —
# enough musical material for Structured-Arrangement to orchestrate, still
# tight enough to find plenty of matches in folk corpora.
N_BARS_WINDOW = 8


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

def _load_midi_aligned(midi_path: str, max_polyphony: int = 4) -> np.ndarray | None:
    """Load a MIDI, align bar 0 to the first downbeat, quantize to 16th-note
    subbeats (using the file's own tempo curve), and return a CP tensor.

    Returns None if the file is not in pure 4/4 — Nottingham contains many 3/4
    waltzes and 6/8 jigs that would otherwise be misinterpreted as 4/4 by the
    BEATS_PER_BAR=4 segmenter."""
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Reject anything that isn't pure 4/4. An empty time_signature_changes list
    # defaults to 4/4 in MIDI semantics, so we accept that.
    ts_changes = pm.time_signature_changes
    for ts in ts_changes:
        if ts.numerator != 4 or ts.denominator != 4:
            return None

    beats     = pm.get_beats()
    downbeats = pm.get_downbeats()
    if len(beats) < 2 or len(downbeats) == 0:
        return None

    t_start_db = downbeats[0]
    # First beat at or after the first downbeat (anchors bar 0 on the downbeat).
    start_beat_idx = int(np.searchsorted(beats, t_start_db - 1e-9, side='left'))
    if start_beat_idx >= len(beats) - 1:
        return None

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

def _extract_bar_roots_from_cp(cp_data: np.ndarray) -> list[int]:
    """Compute bar-level chord root via chromagram on a preprocessed CP tensor.

    cp_data : (n_subbeats, n_voices * 4) uint8 — output of preprocess_midi.
              Each voice slot is [program, pitch, dur_idx, velocity]; padding=255,
              EOS program=254.
    """
    n_subbeats = cp_data.shape[0]
    n_voices   = cp_data.shape[1] // 4
    n_bars     = n_subbeats // SUBBEATS_PER_BAR

    bar_chroma = np.zeros((n_bars, 12), dtype=np.float32)
    progs  = cp_data[:, 0::4]
    pitches = cp_data[:, 1::4]
    for sb in range(n_subbeats):
        bar = sb // SUBBEATS_PER_BAR
        if bar >= n_bars:
            break
        for v in range(n_voices):
            prog  = int(progs[sb, v])
            pitch = int(pitches[sb, v])
            if prog == 255 or prog == 254 or prog == 127:   # padding/EOS/drum
                continue
            if pitch == 255:
                continue
            bar_chroma[bar, pitch % 12] += 1.0

    bar_roots = []
    for b in range(n_bars):
        if bar_chroma[b].sum() < 1e-3:
            bar_roots.append(-1)
            continue
        chroma = bar_chroma[b] / bar_chroma[b].sum()
        best_r, best_s = 0, -1.0
        for r in range(12):
            s = chroma[r] + chroma[(r + 4) % 12] + chroma[(r + 7) % 12]
            if s > best_s:
                best_s, best_r = s, r
        bar_roots.append(best_r)
    return bar_roots


def _find_ivvi_windows(bar_roots: list[int], n_bars: int = 4,
                       phase_zero_only: bool = True) -> list[dict]:
    """Return list of {start_bar, phase, key} dicts for n_bars-bar windows whose
    detected chord roots exactly match the I-IV-V-I rule:
        bar_roots[i+j] == (key + OFFSETS[(j + phase) % 4]) % 12   for all j

    n_bars must be a multiple of 4 (=> integer cycles of I-IV-V-I).
    phase_zero_only=True restricts to windows that start on I (phase=0)."""
    assert n_bars % 4 == 0, f'n_bars must be a multiple of 4, got {n_bars}'
    phases = (0,) if phase_zero_only else (0, 1, 2, 3)
    results = []
    for i in range(len(bar_roots) - n_bars + 1):
        roots = bar_roots[i:i + n_bars]
        if any(r < 0 for r in roots):
            continue
        matched = False
        for phase in phases:
            for key in range(12):
                if all(roots[j] == (key + OFFSETS[(j + phase) % 4]) % 12
                       for j in range(n_bars)):
                    results.append({'start_bar': i, 'phase': phase, 'key': key})
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
# Lead-sheet snippet writer — keeps track separation so the snippet can be
# fed directly to Structured-Arrangement (track 0 = melody, others = chord/
# accompaniment). Re-quantizes notes onto the downbeat-aligned 16th-note
# grid and renders at 120 BPM so timing is clean.
# ---------------------------------------------------------------------------

def _save_leadsheet_snippet(midi_path: str, start_bar: int, n_bars: int,
                            out_path: str, tempo: float = 120.0) -> bool:
    """Save bars [start_bar, start_bar + n_bars) as a multi-track MIDI with
    each original instrument kept as a separate `pretty_midi.Instrument` (so
    track 0 stays melody, track 1 stays chord, etc.)."""
    pm = pretty_midi.PrettyMIDI(midi_path)

    for ts in pm.time_signature_changes:
        if ts.numerator != 4 or ts.denominator != 4:
            return False

    beats     = pm.get_beats()
    downbeats = pm.get_downbeats()
    if len(beats) < 2 or len(downbeats) == 0:
        return False

    start_beat_idx = int(np.searchsorted(beats, downbeats[0] - 1e-9, side='left'))
    if start_beat_idx >= len(beats) - 1:
        return False

    subbeat_times: list[float] = []
    for i in range(start_beat_idx, len(beats) - 1):
        dt = (beats[i + 1] - beats[i]) / BEAT_DIV
        for j in range(BEAT_DIV):
            subbeat_times.append(beats[i] + j * dt)
    subbeat_times.append(float(beats[-1]))
    subbeat_times_arr = np.asarray(subbeat_times)

    start_sb   = start_bar * SUBBEATS_PER_BAR
    n_subbeats = n_bars * SUBBEATS_PER_BAR
    end_sb     = start_sb + n_subbeats
    if end_sb > len(subbeat_times_arr):
        return False

    t_win_start = subbeat_times_arr[start_sb]
    t_win_end   = subbeat_times_arr[end_sb] if end_sb < len(subbeat_times_arr) \
                  else subbeat_times_arr[-1]

    out_pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    sec_per_sb = 60.0 / tempo / BEAT_DIV
    any_note   = False

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            if note.start < t_win_start or note.start >= t_win_end:
                continue
            s_abs = int(np.searchsorted(subbeat_times_arr, note.start, side='right')) - 1
            e_abs = int(np.searchsorted(subbeat_times_arr, note.end,   side='right')) - 1
            s_loc = s_abs - start_sb
            e_loc = e_abs - start_sb
            if s_loc < 0 or s_loc >= n_subbeats:
                continue
            e_loc = max(s_loc + 1, min(e_loc, n_subbeats))
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
        os.makedirs(args.save_examples_dir, exist_ok=True)

    for midi_path in files:
        try:
            cp_data = _load_midi_aligned(midi_path, max_polyphony=args.max_polyphony)
        except Exception as e:
            n_skipped += 1
            print(f'  SKIP {os.path.basename(midi_path)}: {e}')
            continue

        if cp_data is None:
            n_skipped += 1
            continue

        bar_roots = _extract_bar_roots_from_cp(cp_data)
        windows   = _find_ivvi_windows(bar_roots, n_bars=args.n_bars,
                                       phase_zero_only=not args.all_phases)
        for w in windows:
            manifest.append({
                'midi_path':  midi_path,
                'start_bar':  w['start_bar'],
                'phase':      w['phase'],
                'key':        w['key'],
                'key_name':   ROOT_NAMES[w['key']],
                'n_bars':     args.n_bars,
                'bar_roots':  bar_roots[w['start_bar']:w['start_bar'] + args.n_bars],
            })
            if args.save_examples_dir and examples_per_key[w['key']] < args.examples_per_key:
                stem = os.path.splitext(os.path.basename(midi_path))[0]
                out  = os.path.join(
                    args.save_examples_dir,
                    f'key{ROOT_NAMES[w["key"]]}_phase{w["phase"]}_bar{w["start_bar"]:04d}_{stem}.mid',
                )
                try:
                    if _save_midi_example(cp_data, w['start_bar'], args.n_bars,
                                          args.max_polyphony, out):
                        examples_per_key[w['key']] += 1
                except Exception as e:
                    print(f'  example-save failed for {midi_path}: {e}')
        n_windows += len(windows)

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
    print(f'Window length: {n_bars_window} bars ({n_subbeats_window} subbeats)')

    for entry in manifest:
        midi_path = entry['midi_path']
        if midi_path not in cp_cache:
            try:
                cp_arr = _load_midi_aligned(midi_path, max_polyphony=args.max_polyphony)
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
            for sb in range(n_subbeats_window):
                bar_in_window = sb // SUBBEATS_PER_BAR
                root = (target_key + OFFSETS[(bar_in_window + phase) % 4]) % 12
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
        stem      = os.path.splitext(os.path.basename(midi_path))[0]
        out       = os.path.join(
            args.out_dir,
            f'{n_saved:06d}_key{ROOT_NAMES[key]}_phase{phase}_bar{start_bar:04d}_{stem}.mid',
        )
        try:
            if _save_leadsheet_snippet(midi_path, start_bar, n_bars, out):
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
                    help='Window length in bars; must be a multiple of 4. '
                         'Default 8 (= 2 I-IV-V-I cycles, 128 subbeats).')
    pf.add_argument('--all_phases',    action='store_true',
                    help='Accept all 4 cyclic rotations (start on I/IV/V/I). '
                         'Default keeps only phase 0 (windows start on I).')
    pf.add_argument('--save_examples_dir', type=str, default=None,
                    help='If set, save MIDI snippets of qualifying windows here '
                         '(uses original-tempo PrettyMIDI so playback sounds natural).')
    pf.add_argument('--examples_per_key',  type=int, default=3,
                    help='Cap on saved examples per detected key (default 3).')

    pe = sub.add_parser('extract', help='Extract CP tensors from a manifest')
    pe.add_argument('--manifest',       required=True)
    pe.add_argument('--out_dir',        required=True)
    pe.add_argument('--out_pt',         required=True)
    pe.add_argument('--max_polyphony',  type=int, default=4)
    pe.add_argument('--min_notes_per_bar', type=int, default=2)
    pe.add_argument('--target_keys',    type=int, nargs='+', default=list(SEEN_KEYS_DEFAULT),
                    help='Pitch classes (0-11) to transpose every window into. Each '
                         'original window is duplicated len(target_keys) times, one per '
                         'target key (smallest signed shift). Default: all 12 keys '
                         'except F# (6) and G# (8), reserved as unseen-eval keys. '
                         'Pass "6 8" to build the unseen-eval dataset.')

    pl = sub.add_parser('leadsheets',
                        help='Save matched 4-bar windows as multi-track MIDI lead '
                             'sheets (track 0 = melody, track 1 = chord, …) ready '
                             'for Structured-Arrangement Stage 1/2 input.')
    pl.add_argument('--manifest',   required=True)
    pl.add_argument('--out_dir',    required=True)
    pl.add_argument('--limit',      type=int, default=0,
                    help='Cap total snippets saved (0 = no cap).')
    pl.add_argument('--per_key',    type=int, default=0,
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
