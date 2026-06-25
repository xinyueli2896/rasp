"""
Two-step pipeline for extracting I-IV-V-I windows from a MIDI corpus.

Step 1 — filter (fast, no tensor computation):
  Scan all MIDI files, extract bar-level chord roots via chromagram analysis,
  find every 4-bar window matching I-IV-V-I (any key, any cyclic rotation),
  save a JSON manifest of matching windows.

  python -m midi_adapter.filter_nottingham filter \\
      --input_dir data/nottingham/MIDI \\
      --manifest  data/nottingham_ivvi_manifest.json

Step 2 — extract (slower, computes CP tensors):
  Read the manifest, extract CP tensors for each window, save dataset files
  in the same format as generate_synthetic_bass.py.

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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.generate_synthetic_bass import (
    _preprocess_pm, pitch_sort_cp,
    BEAT_DIV, SUBBEATS_PER_BAR, BEATS_PER_BAR, OFFSETS,
    DURATION_BOUNDARIES,
)

try:
    import pretty_midi
except ImportError:
    print('pretty_midi is required: pip install pretty_midi')
    sys.exit(1)

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ---------------------------------------------------------------------------
# I-IV-V-I pattern matching
# ---------------------------------------------------------------------------

# Consecutive root-diff signatures for all 4 cyclic rotations of I-IV-V-I
_IVVI_ROTATIONS = [
    (0, (5, 2, 5)),   # I  → IV → V  → I
    (1, (2, 5, 0)),   # IV → V  → I  → I
    (2, (5, 0, 5)),   # V  → I  → I  → IV
    (3, (0, 5, 2)),   # I  → I  → IV → V
]

# For each phase, the key (I-chord root) relative to the window's first bar root
_PHASE_KEY_OFFSET = {0: 0, 1: 7, 2: 5, 3: 0}


def _find_ivvi_windows(bar_roots: list[int]) -> list[dict]:
    """Return list of {start_bar, phase, key} dicts for all matching 4-bar windows."""
    results = []
    for i in range(len(bar_roots) - 3):
        roots = bar_roots[i:i + 4]
        if any(r < 0 for r in roots):
            continue
        diffs = tuple((roots[j + 1] - roots[j]) % 12 for j in range(3))
        for phase, sig in _IVVI_ROTATIONS:
            if diffs == sig:
                key = (roots[0] - _PHASE_KEY_OFFSET[phase]) % 12
                results.append({'start_bar': i, 'phase': phase, 'key': key})
    return results


# ---------------------------------------------------------------------------
# Step 1: Filter — chromagram-based chord extraction + window finding
# ---------------------------------------------------------------------------

def _extract_bar_roots(midi_path: str, beats_per_bar: int = BEATS_PER_BAR) -> list[int]:
    """Return bar-level chord root (0-11) via chromagram major-triad matching.
    Returns -1 for bars with no notes."""
    pm         = pretty_midi.PrettyMIDI(midi_path)
    beat_times = pm.get_beats()
    n_beats    = len(beat_times)
    n_bars     = n_beats // beats_per_bar

    beat_chroma = np.zeros((n_beats, 12), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            s = int(np.searchsorted(beat_times, note.start, side='right')) - 1
            e = int(np.searchsorted(beat_times, note.end,   side='right')) - 1
            s = max(0, min(s, n_beats - 1))
            for b in range(s, min(e + 1, n_beats)):
                beat_chroma[b, note.pitch % 12] += 1.0

    bar_roots = []
    for b in range(n_bars):
        chroma = beat_chroma[b * beats_per_bar:(b + 1) * beats_per_bar].sum(axis=0)
        if chroma.sum() < 1e-3:
            bar_roots.append(-1)
        else:
            chroma /= chroma.sum()
            best_r, best_s = 0, -1.0
            for r in range(12):
                s = chroma[r] + chroma[(r + 4) % 12] + chroma[(r + 7) % 12]
                if s > best_s:
                    best_s, best_r = s, r
            bar_roots.append(best_r)
    return bar_roots


def cmd_filter(args):
    exts  = {'.mid', '.midi', '.MID', '.MIDI'}
    files = sorted([
        os.path.join(root, f)
        for root, _, fnames in os.walk(args.input_dir)
        for f in fnames if os.path.splitext(f)[1] in exts
    ])
    print(f'Scanning {len(files)} MIDI files in {args.input_dir} ...')

    manifest = []
    n_windows = 0

    for midi_path in files:
        try:
            bar_roots = _extract_bar_roots(midi_path)
        except Exception as e:
            print(f'  SKIP {os.path.basename(midi_path)}: {e}')
            continue

        windows = _find_ivvi_windows(bar_roots)
        for w in windows:
            manifest.append({
                'midi_path':  midi_path,
                'start_bar':  w['start_bar'],
                'phase':      w['phase'],
                'key':        w['key'],
                'key_name':   ROOT_NAMES[w['key']],
                'bar_roots':  bar_roots[w['start_bar']:w['start_bar'] + 4],
            })
        if windows:
            n_windows += len(windows)

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nFound {n_windows} windows across {len(files)} files.')
    print(f'Manifest saved → {args.manifest}')
    print(f'\nKey distribution:')
    from collections import Counter
    key_counts = Counter(e['key_name'] for e in manifest)
    for k, v in sorted(key_counts.items(), key=lambda x: -x[1]):
        print(f'  {k:<4} {v}')


# ---------------------------------------------------------------------------
# Step 2: Extract — CP tensors from manifest
# ---------------------------------------------------------------------------

def _extract_window_cp(
    pm: pretty_midi.PrettyMIDI,
    start_bar: int,
    n_bars: int = 4,
    max_polyphony: int = 4,
    beats_per_bar: int = BEATS_PER_BAR,
) -> torch.Tensor | None:
    beat_times = pm.get_beats()
    n_beats_total = len(beat_times)

    # Build 16th-note subbeat times by linearly interpolating within each beat
    subbeat_times = []
    for i in range(n_beats_total - 1):
        dt = (beat_times[i + 1] - beat_times[i]) / BEAT_DIV
        for j in range(BEAT_DIV):
            subbeat_times.append(beat_times[i] + j * dt)
    subbeat_times.append(beat_times[-1])
    subbeat_times = np.array(subbeat_times)

    start_subbeat = start_bar * beats_per_bar * BEAT_DIV   # = start_bar * SUBBEATS_PER_BAR
    n_subbeats    = n_bars * beats_per_bar * BEAT_DIV       # = n_bars   * SUBBEATS_PER_BAR
    end_subbeat   = start_subbeat + n_subbeats

    if end_subbeat > len(subbeat_times):
        return None

    rolls  = np.full((n_subbeats, max_polyphony, 4), 255, dtype=np.uint8)
    counts = np.zeros(n_subbeats, dtype=np.uint8)

    t0 = subbeat_times[start_subbeat]
    t1 = subbeat_times[end_subbeat] if end_subbeat < len(subbeat_times) else pm.get_end_time()

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.start < t0 or note.start >= t1:
                continue
            abs_sub = int(np.searchsorted(subbeat_times, note.start, side='right')) - 1
            local_b = abs_sub - start_subbeat
            if local_b < 0 or local_b >= n_subbeats or counts[local_b] >= max_polyphony:
                continue
            abs_e   = int(np.searchsorted(subbeat_times, note.end, side='right')) - 1
            dur_idx = int(np.searchsorted(DURATION_BOUNDARIES, max(0, abs_e - abs_sub)))
            rolls[local_b, counts[local_b]] = [inst.program, note.pitch, dur_idx, note.velocity]
            counts[local_b] += 1

    for i in range(n_subbeats):
        if counts[i] < max_polyphony:
            rolls[i, counts[i], 0] = 254

    if counts.sum() == 0:
        return None

    data = torch.tensor(rolls.reshape(n_subbeats, max_polyphony * 4), dtype=torch.uint8)
    return pitch_sort_cp(data)


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

    # Cache open PrettyMIDI objects to avoid re-loading the same file repeatedly
    pm_cache: dict[str, pretty_midi.PrettyMIDI] = {}

    for entry in manifest:
        midi_path = entry['midi_path']
        if midi_path not in pm_cache:
            try:
                pm_cache[midi_path] = pretty_midi.PrettyMIDI(midi_path)
            except Exception as e:
                print(f'  SKIP {midi_path}: {e}')
                continue
            # Evict cache if too large
            if len(pm_cache) > 200:
                oldest = next(iter(pm_cache))
                del pm_cache[oldest]

        pm        = pm_cache[midi_path]
        start_bar = entry['start_bar']
        key       = entry['key']
        phase     = entry['phase']

        data = _extract_window_cp(pm, start_bar, n_bars=args.n_bars,
                                  max_polyphony=args.max_polyphony)
        if data is None:
            continue

        n_notes = (data[:, 0::4] != 255).sum().item()
        if n_notes < args.n_bars * args.min_notes_per_bar:
            continue

        # Subbeat-level chord tokens for this window (one per 16th note)
        n_subbeats_window = args.n_bars * SUBBEATS_PER_BAR
        beat_tokens = []
        for sb in range(n_subbeats_window):
            bar_in_window = sb // SUBBEATS_PER_BAR
            root = (key + OFFSETS[(bar_in_window + phase) % 4]) % 12
            beat_tokens.append(chord_str_to_token(f'{ROOT_NAMES[root]}:maj'))
        all_chords.append(torch.tensor(beat_tokens, dtype=torch.int16))

        rel = f'nottingham_{n_saved:06d}_key{key}_phase{phase}.mid'
        txt_lines.append(f'{n_saved}\t{rel}')
        all_data.append(data)
        n_saved += 1

        if n_saved % 200 == 0:
            print(f'  {n_saved} windows extracted ...')

    if n_saved == 0:
        print('No windows passed the note-density filter. Try --min_notes_per_bar 1')
        return

    n_subbeats = args.n_bars * SUBBEATS_PER_BAR
    torch.save(torch.cat(all_data, dim=0),                f'{args.out_pt}.pt')
    torch.save(torch.tensor([n_subbeats] * n_saved),      f'{args.out_pt}.length.pt')
    torch.save(torch.zeros(n_saved, 2, dtype=torch.int8), f'{args.out_pt}.pitch_shift_range.pt')
    torch.save(all_chords,                                 f'{args.out_pt}.beat_chords.pt')
    with open(f'{args.out_pt}.txt', 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    print(f'\nSaved {n_saved} windows.')
    print(f'Dataset → {args.out_pt}.{{pt,length.pt,pitch_shift_range.pt,beat_chords.pt,txt}}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Filter MIDI corpus for I-IV-V-I windows')
    sub = p.add_subparsers(dest='cmd', required=True)

    # --- filter ---
    pf = sub.add_parser('filter', help='Scan MIDI files and save matching-window manifest')
    pf.add_argument('--input_dir', required=True,
                    help='Directory of MIDI files (searched recursively)')
    pf.add_argument('--manifest',  required=True,
                    help='Output JSON manifest path')

    # --- extract ---
    pe = sub.add_parser('extract', help='Extract CP tensors from a manifest')
    pe.add_argument('--manifest',       required=True)
    pe.add_argument('--out_dir',        required=True,
                    help='Directory for MIDI snippets (for inspection)')
    pe.add_argument('--out_pt',         required=True,
                    help='Output prefix for .pt dataset files')
    pe.add_argument('--n_bars',         type=int, default=4)
    pe.add_argument('--max_polyphony',  type=int, default=4)
    pe.add_argument('--min_notes_per_bar', type=int, default=2)

    args = p.parse_args()
    if args.cmd == 'filter':
        cmd_filter(args)
    else:
        cmd_extract(args)


if __name__ == '__main__':
    main()
