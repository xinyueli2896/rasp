"""
Pre-transpose a leadsheets directory into every target key BEFORE feeding to
AccoMontage. This is the input side of Option B (transpose → orchestrate)
where each transposed snippet gets its own independent orchestration.

Input filenames must include a `_keyXX_` tag (as produced by
`filter_nottingham leadsheets`), e.g.
    000123_keyC_phase0_bar0044_049.mid

For each input we compute the signed semitone shift to reach every
--target_keys entry, apply the shift to every non-drum track, and write the
output as
    <output_dir>/000123_keyNEWKEY_phase0_bar0044_049.mid

The output can then be fed to batch_orchestrate.py exactly as if it were a
larger leadsheets directory — one snippet per input × target_key pair.

Usage
-----
    python -m midi_adapter.transpose_leadsheets \\
        --in_dir      /l/users/xinyue.li/data/pop909_ivvi_w1_snippets \\
        --out_dir     /l/users/xinyue.li/data/pop909_ivvi_w1_snippets_perkey \\
        --target_keys 0 1 2 3 4 5 7 9 10 11 6 8   # 10 seen + 2 unseen
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.filter_nottingham import (
    _signed_shift,
    ROOT_NAMES,
    SEEN_KEYS_DEFAULT,
)


_KEY_RE = re.compile(r'_key([A-G]#?)_')
_KEY_STR_TO_PC = {n: i for i, n in enumerate(ROOT_NAMES)}


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
    p.add_argument('--in_dir',      required=True,
                   help='Leadsheets directory produced by '
                        '`filter_nottingham leadsheets`. Filenames must contain '
                        '_keyXX_ tags.')
    p.add_argument('--out_dir',     required=True,
                   help='Where to write the pre-transposed leadsheets.')
    p.add_argument('--target_keys', type=int, nargs='+',
                   default=list(SEEN_KEYS_DEFAULT) + [6, 8],
                   help='Pitch classes to transpose each snippet into. Default '
                        '= 10 seen + F# + G# (both unseen), giving 12 keys.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.in_dir)
                   if f.lower().endswith(('.mid', '.midi')))
    print(f'Found {len(files)} leadsheets in {args.in_dir}')

    n_written = 0
    n_skipped = 0
    for fname in files:
        m = _KEY_RE.search(fname)
        if m is None or m.group(1) not in _KEY_STR_TO_PC:
            n_skipped += 1
            continue
        base_key = _KEY_STR_TO_PC[m.group(1)]

        src = os.path.join(args.in_dir, fname)
        try:
            pm_src = pretty_midi.PrettyMIDI(src)
        except Exception as e:
            print(f'  SKIP {fname}: cannot load — {e}')
            n_skipped += 1
            continue

        for target_key in args.target_keys:
            shift = _signed_shift(target_key, base_key)
            new_name = _KEY_RE.sub(f'_key{ROOT_NAMES[target_key]}_', fname)
            if new_name == fname and target_key != base_key:
                # Regex miss (defensive) — put a tag on the filename anyway
                new_name = f'{ROOT_NAMES[target_key]}__{fname}'
            out_path = os.path.join(args.out_dir, new_name)
            try:
                pm_new = pretty_midi.PrettyMIDI(src)
                _transpose_midi_in_place(pm_new, shift)
                pm_new.write(out_path)
                n_written += 1
            except Exception as e:
                print(f'  FAIL {fname} → key {ROOT_NAMES[target_key]}: {e}')

    print(f'\nWrote {n_written} transposed leadsheets to {args.out_dir}')
    print(f'({n_skipped} inputs skipped)')


if __name__ == '__main__':
    main()
