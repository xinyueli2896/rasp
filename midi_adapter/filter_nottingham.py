"""
Two-step pipeline for extracting I-IV-V-I windows from a MIDI corpus.

Uses `preprocess_midi` from preprocess_large_midi_dataset.py (which uses
`xf_midi.XFMidi` with constant_tempo=60/beat_div) so the resulting CP tensor
matches the LA-pretrained CP transformer exactly. Chord detection is done
directly on the loaded CP tensor (no second pass over the raw MIDI).

Step 1 — filter (loads MIDIs via xf_midi, finds I-IV-V-I windows):
  python -m midi_adapter.filter_nottingham filter \\
      --input_dir data/nottingham/MIDI \\
      --manifest  data/nottingham_ivvi_manifest.json

Step 2 — extract (slices the cached CP tensors per window):
  python -m midi_adapter.filter_nottingham extract \\
      --manifest  data/nottingham_ivvi_manifest.json \\
      --out_dir   data/nottingham_ivvi_midi \\
      --out_pt    data/nottingham_ivvi_cp4

Requires `preprocess_large_midi_dataset.py` and `xf_midi.py` on PYTHONPATH
(copy them from the midi_yinyang repo into the project root).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess_large_midi_dataset import preprocess_midi   # uses xf_midi, beat_div=4

from midi_adapter.generate_synthetic_bass import (
    pitch_sort_cp,
    BEAT_DIV, SUBBEATS_PER_BAR, BEATS_PER_BAR, OFFSETS,
)

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


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
# MIDI snippet writer — slice the original (variable-tempo) PrettyMIDI so the
# saved example sounds at the file's natural tempo (not the constant-tempo
# UglyMIDI version used for chord detection).
# ---------------------------------------------------------------------------

def _save_midi_example(midi_path: str, start_bar: int, n_bars: int,
                      out_path: str) -> bool:
    """Slice bars [start_bar, start_bar + n_bars) from the original MIDI and
    write to out_path. Returns True if a non-empty snippet was written."""
    pm         = pretty_midi.PrettyMIDI(midi_path)
    beat_times = pm.get_beats()
    start_beat = start_bar * BEATS_PER_BAR
    end_beat   = start_beat + n_bars * BEATS_PER_BAR
    if end_beat > len(beat_times):
        return False
    t0 = beat_times[start_beat]
    t1 = beat_times[end_beat] if end_beat < len(beat_times) else pm.get_end_time()

    out_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum)
        for note in inst.notes:
            if note.start < t0 or note.start >= t1:
                continue
            new_inst.notes.append(pretty_midi.Note(
                velocity = note.velocity,
                pitch    = note.pitch,
                start    = note.start - t0,
                end      = min(note.end, t1) - t0,
            ))
        if new_inst.notes:
            out_pm.instruments.append(new_inst)

    if not out_pm.instruments:
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
            result = preprocess_midi(
                midi_path,
                max_polyphony = args.max_polyphony,
                beat_div      = BEAT_DIV,
                ins_ids       = 'all',
                filter        = False,   # Nottingham is well-quantized; skip LA heuristic
                dedup         = False,
            )
        except Exception as e:
            n_skipped += 1
            print(f'  SKIP {os.path.basename(midi_path)}: {e}')
            continue

        if result is None:
            n_skipped += 1
            continue

        cp_tensor, _ = result
        cp_data = cp_tensor.numpy()
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
                    if _save_midi_example(midi_path, w['start_bar'], args.n_bars, out):
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

    # Use n_bars from the manifest entries if present (filter step stamps it).
    # All entries in a manifest share the same n_bars, so just read the first.
    n_bars_window = manifest[0].get('n_bars', args.n_bars) if manifest else args.n_bars
    if n_bars_window != args.n_bars:
        print(f'  manifest n_bars={n_bars_window} overrides --n_bars={args.n_bars}')
    n_subbeats_window = n_bars_window * SUBBEATS_PER_BAR

    for entry in manifest:
        midi_path = entry['midi_path']
        if midi_path not in cp_cache:
            try:
                result = preprocess_midi(
                    midi_path,
                    max_polyphony = args.max_polyphony,
                    beat_div      = BEAT_DIV,
                    ins_ids       = 'all',
                    filter        = False,
                    dedup         = False,
                )
            except Exception as e:
                print(f'  SKIP {midi_path}: {e}')
                cp_cache[midi_path] = None
                continue
            if result is None:
                cp_cache[midi_path] = None
                continue
            cp_cache[midi_path] = result[0].numpy()
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
    pf.add_argument('--n_bars',        type=int, default=4,
                    help='Window length in bars; must be a multiple of 4 (= integer '
                         'I-IV-V-I cycles). 4 → 64 subbeats, 16 → 256, 24 → 384.')
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
    pe.add_argument('--n_bars',         type=int, default=4)
    pe.add_argument('--max_polyphony',  type=int, default=4)
    pe.add_argument('--min_notes_per_bar', type=int, default=2)
    pe.add_argument('--target_keys',    type=int, nargs='+', default=list(SEEN_KEYS_DEFAULT),
                    help='Pitch classes (0-11) to transpose every window into. Each '
                         'original window is duplicated len(target_keys) times, one per '
                         'target key (smallest signed shift). Default: all 12 keys '
                         'except F# (6) and G# (8), reserved as unseen-eval keys. '
                         'Pass "6 8" to build the unseen-eval dataset.')

    args = p.parse_args()
    if args.cmd == 'filter':
        cmd_filter(args)
    else:
        cmd_extract(args)


if __name__ == '__main__':
    main()
