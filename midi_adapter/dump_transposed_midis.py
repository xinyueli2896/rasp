"""
Render a source window at MULTIPLE target keys so you can play the transposed
MIDIs side by side and hear whether the pitch-shift step is correct.

For each manifest entry (up to --per_source_song per unique source MIDI):
  1. Load the source MIDI (multi-track).
  2. Slice bars [start_bar, start_bar + n_bars) with downbeat alignment.
  3. Quantize onto the 16th-note grid at 120 BPM.
  4. Apply repair to `wrong_positions` (per-track, same policy as the training
     extract).
  5. For each requested target key, transpose every non-drum note in every
     track by the smallest signed shift `(target_key − source_key) mod 12`
     mapped into [-5, +6], and write one MIDI per (source, target_key).

Output layout — one subfolder per source, one MIDI per target key inside so
you can A/B them:
    out_dir/<source_stem>_<bar>_<sourceKey>_wrong<N>/
        key_C.mid
        key_D.mid
        key_F#.mid
        ...

Usage
-----
    python -m midi_adapter.dump_transposed_midis \\
        --manifest    /l/users/xinyue.li/data/pop909_ivvi_w1_manifest.json \\
        --out_dir     /l/users/xinyue.li/data/pop909_transposed_check \\
        --target_keys 0 5 7 6 8 \\
        --per_source_song 2 \\
        --no_align
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.filter_nottingham import (
    _save_repaired_snippet_original_tracks,
    _signed_shift,
    ROOT_NAMES,
)


def _transpose_midi_in_place(pm: pretty_midi.PrettyMIDI, shift: int) -> None:
    if shift == 0:
        return
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            new_pitch = note.pitch + shift
            note.pitch = max(0, min(127, new_pitch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',       required=True)
    p.add_argument('--out_dir',        required=True)
    p.add_argument('--target_keys',    type=int, nargs='+',
                   default=[0, 2, 5, 7, 9, 6, 8],
                   help='Pitch classes to transpose into. Default includes a couple '
                        'of seen + F# and G# so you can hear the unseen-key transposition too.')
    p.add_argument('--per_source_song', type=int, default=2,
                   help='Cap manifest entries picked per unique source MIDI.')
    p.add_argument('--no_align',       action='store_true',
                   help='Match the flag used at filter time (POP909 → set; Nottingham → omit).')
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f'Loaded {len(manifest)} entries from {args.manifest}')
    os.makedirs(args.out_dir, exist_ok=True)

    per_source_count: dict[str, int] = {}
    n_written = 0

    for entry in manifest:
        midi_path = entry['midi_path']
        if per_source_count.get(midi_path, 0) >= args.per_source_song:
            continue
        per_source_count[midi_path] = per_source_count.get(midi_path, 0) + 1

        start_bar       = entry['start_bar']
        n_bars          = entry.get('n_bars', 4)
        chords_per_bar  = entry.get('chords_per_bar', 2)
        base_key        = entry['key']
        phase           = entry['phase']
        wrong_positions = entry.get('wrong_positions', [])

        stem     = os.path.splitext(os.path.basename(midi_path))[0]
        wrong_tag = f'_wrong{len(wrong_positions)}' if wrong_positions else ''
        folder = os.path.join(
            args.out_dir,
            f'{stem}_bar{start_bar:04d}_key{ROOT_NAMES[base_key]}{wrong_tag}',
        )
        os.makedirs(folder, exist_ok=True)

        # Write the source-key (untransposed, repaired) snippet as a reference.
        ref_path = os.path.join(folder, f'ref_key{ROOT_NAMES[base_key]}.mid')
        try:
            wrote = _save_repaired_snippet_original_tracks(
                midi_path, start_bar, n_bars, chords_per_bar,
                base_key, phase, wrong_positions,
                out_path=ref_path,
                align_to_downbeat=not args.no_align,
            )
            if not wrote:
                print(f'  SKIP {midi_path}: reference render failed')
                continue
        except Exception as e:
            print(f'  SKIP {midi_path}: {e}')
            continue

        # Then load the reference and transpose it for each target key.
        for target_key in args.target_keys:
            shift = _signed_shift(target_key, base_key)
            out_path = os.path.join(folder, f'key_{ROOT_NAMES[target_key]}.mid')
            try:
                pm = pretty_midi.PrettyMIDI(ref_path)
                _transpose_midi_in_place(pm, shift)
                pm.write(out_path)
                n_written += 1
            except Exception as e:
                print(f'  transpose fail ({stem}, key {ROOT_NAMES[target_key]}): {e}')

    print(f'\nWrote {n_written} transposed MIDIs to {args.out_dir}')
    print(f'({len(per_source_count)} source songs, up to {args.per_source_song} entries each)')


if __name__ == '__main__':
    main()
