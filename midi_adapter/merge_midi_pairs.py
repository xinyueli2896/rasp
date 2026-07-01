"""
Merge two parallel MIDI datasets (chord-only + melody/acc multi-track) into a
single multi-track MIDI per song. Filename stems must match between the two
directories.

Typical POP909-derived case:
  chord_dir/     songs rendered from the chord annotation (1 track)
  melacc_dir/    original POP909 MIDIs (melody + bridge + piano tracks)
  out_dir/       combined: melody + bridge + piano + chord tracks

Usage
-----
  python -m midi_adapter.merge_midi_pairs \\
      --chord_dir   /path/to/pop909_chord_midis \\
      --melacc_dir  /path/to/pop909_melacc_midis \\
      --out_dir     /path/to/pop909_combined_midis

Every instrument keeps its original program, name, drum flag, and notes
on both sides. The chord MIDI's tracks are appended after the melody/acc
MIDI's tracks so the combined file has all of them side-by-side. Tempo
and time-signature metadata are inherited from the melody/acc MIDI.
"""

from __future__ import annotations

import argparse
import os

import pretty_midi


def _midis_by_stem(root: str) -> dict[str, str]:
    """Return {basename_without_ext: absolute_path} for every .mid[i] under root."""
    out: dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(('.mid', '.midi')):
                stem = os.path.splitext(fn)[0]
                out[stem] = os.path.join(dirpath, fn)
    return out


def merge_pair(chord_path: str, melacc_path: str, out_path: str) -> bool:
    """Append the chord MIDI's instrument(s) to the melody/acc MIDI and save.
    Programs, names, drum flags, and note contents are preserved exactly for
    every instrument on both sides — the melody/acc MIDI keeps its full track
    structure, and the chord MIDI's tracks are appended after them as-is."""
    chord_pm  = pretty_midi.PrettyMIDI(chord_path)
    melacc_pm = pretty_midi.PrettyMIDI(melacc_path)

    if not chord_pm.instruments:
        return False
    melacc_pm.instruments.extend(chord_pm.instruments)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    melacc_pm.write(out_path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chord_dir',    required=True,
                   help='Folder of chord-only MIDIs')
    p.add_argument('--melacc_dir',   required=True,
                   help='Folder of melody + accompaniment MIDIs (multi-track)')
    p.add_argument('--out_dir',      required=True,
                   help='Where to write combined MIDIs')
    p.add_argument('--limit',        type=int, default=0,
                   help='Cap total files (0 = no cap)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    chords = _midis_by_stem(args.chord_dir)
    melaccs = _midis_by_stem(args.melacc_dir)
    common = sorted(set(chords) & set(melaccs))

    if args.limit:
        common = common[:args.limit]

    print(f'chord_dir:  {len(chords)} MIDIs found')
    print(f'melacc_dir: {len(melaccs)} MIDIs found')
    print(f'Matching stems: {len(common)}')
    missing_chord  = sorted(set(melaccs) - set(chords))[:5]
    missing_melacc = sorted(set(chords)  - set(melaccs))[:5]
    if missing_chord:
        print(f'  {len(set(melaccs) - set(chords))} melacc without chord (first 5): {missing_chord}')
    if missing_melacc:
        print(f'  {len(set(chords) - set(melaccs))} chord without melacc (first 5): {missing_melacc}')

    n_ok = 0
    for stem in common:
        out = os.path.join(args.out_dir, f'{stem}.mid')
        try:
            if merge_pair(chords[stem], melaccs[stem], out):
                n_ok += 1
        except Exception as e:
            print(f'  SKIP {stem}: {e}')

    print(f'\nMerged {n_ok}/{len(common)} songs → {args.out_dir}')


if __name__ == '__main__':
    main()
