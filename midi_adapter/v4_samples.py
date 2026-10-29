"""
Listening samples from the v4 stage: MIDI plus a chord annotation per span.

For each selected span this writes one folder holding

    bass.mid     the bass span alone -- the thing the model trains on
    full.mid     every track of the source file over the same bars, so the
                 bass can be heard in its arrangement
    chords.lab   one line per bar, `start  end  label` in seconds, Harte-style
                 (C:maj, A:min, D) -- loads into Sonic Visualiser and friends
    chords.txt   the same, readable: bar, time, root the rule names, the bass
                 note actually struck, and the quality guess

The ROOT is exact: it is what the bass strikes on the downbeat, which is what
the scan selected the span for, so it equals (key + OFFSETS[bar % 4]) % 12 by
construction. The QUALITY is a guess: the scan never looked at it, so it is
read here from the other tracks -- whichever of the major or minor third
carries more duration in the bar. A bar where neither sounds gets the bare
root. It is annotation for the ear, not ground truth the rule depends on.

Times are written at 120 BPM (one subbeat = 0.125 s), which is arbitrary --
the scan's grid is in subbeats and carries no tempo.

Usage
-----
  python -m midi_adapter.v4_samples \\
      --stage_dir ../stage --midi_zip ../LAMD/<archive>.zip \\
      --upstream ../mald-upstream --out_dir ../samples --per_key 2
"""

from __future__ import annotations

import argparse
import collections
import io
import os
import zipfile

import numpy as np

from midi_adapter.survey_mald import OFFSETS, ROOT_NAMES
from midi_adapter.prepare_v4_data import (
    BAR_SUBBEATS, BEAT_DIV, bass_track, load_upstream, stage_get,
)

TEMPO = 120.0
SEC_PER_SUBBEAT = 60.0 / TEMPO / BEAT_DIV          # 0.125 s


def _cut(ins, lo: int, hi: int, pretty_midi):
    """Copy of one instrument restricted to notes starting in [lo, hi)."""
    out = pretty_midi.Instrument(program=ins.program, is_drum=ins.is_drum,
                                 name=ins.name)
    for n in ins.notes:
        s = int(round(n.start))
        if lo <= s < hi:
            e = min(max(s + 1, int(round(n.end))), hi)
            out.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=(s - lo) * SEC_PER_SUBBEAT, end=(e - lo) * SEC_PER_SUBBEAT))
    return out


def _quality(midi, bass, lo: int, hi: int, root: int) -> str:
    """maj / min / '' from duration-weighted pitch classes of the other tracks."""
    w = np.zeros(12)
    for ins in midi.instruments:
        if ins.is_drum or ins is bass:
            continue
        for n in ins.notes:
            s, e = max(int(round(n.start)), lo), min(int(round(n.end)), hi)
            if e > s:
                w[n.pitch % 12] += e - s
    maj, mnr = w[(root + 4) % 12], w[(root + 3) % 12]
    if maj == 0 and mnr == 0:
        return ''
    return 'maj' if maj >= mnr else 'min'


def write_sample(span: dict, zf, upstream: str, out_dir: str) -> str | None:
    import pretty_midi
    UglyMIDI, _, _ = load_upstream(upstream)
    try:
        midi = UglyMIDI(io.BytesIO(zf.read(span['path'])),
                        constant_tempo=60.0 / BEAT_DIV)
    except Exception:                                          # noqa: BLE001
        return None
    bass = bass_track(midi)
    if bass is None:
        return None
    key, n_bars = span['key'], span['n_bars']
    lo = span['start_subbeat']
    hi = lo + n_bars * BAR_SUBBEATS
    stem = os.path.splitext(os.path.basename(span['path']))[0][:8]
    name = (f'key{key:02d}_{ROOT_NAMES[key].replace("#", "s")}_{n_bars:02d}bars'
            f'_{stem}_bar{span["start_bar"]}')
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    pm.instruments.append(_cut(bass, lo, hi, pretty_midi))
    pm.write(os.path.join(d, 'bass.mid'))

    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    for ins in midi.instruments:
        cut = _cut(ins, lo, hi, pretty_midi)
        if cut.notes:
            pm.instruments.append(cut)
    pm.write(os.path.join(d, 'full.mid'))

    onsets = collections.defaultdict(list)
    for n in bass.notes:
        onsets[int(round(n.start))].append(n.pitch)
    rows, labels = [], []
    for b in range(n_bars):
        blo = lo + b * BAR_SUBBEATS
        root = (key + OFFSETS[b % 4]) % 12
        struck = min(onsets.get(blo, [None]) or [None]) if onsets.get(blo) else None
        q = _quality(midi, bass, blo, blo + BAR_SUBBEATS, root)
        label = f'{ROOT_NAMES[root]}:{q}' if q else ROOT_NAMES[root]
        labels.append(label)
        t0 = b * BAR_SUBBEATS * SEC_PER_SUBBEAT
        rows.append((b, t0, root, struck, label))
    with open(os.path.join(d, 'chords.lab'), 'w') as f:
        for b, t0, _, _, label in rows:
            f.write(f'{t0:.3f}\t{t0 + BAR_SUBBEATS * SEC_PER_SUBBEAT:.3f}\t{label}\n')
    with open(os.path.join(d, 'chords.txt'), 'w') as f:
        f.write(f'source   : {span["path"]}\n')
        f.write(f'span     : bars {span["start_bar"]}..{span["start_bar"] + n_bars - 1} '
                f'of the source ({n_bars} bars, {n_bars // 4} cycles)\n')
        f.write(f'key      : {ROOT_NAMES[key]} (key {key})   '
                f'rule: root[bar] = (key + {OFFSETS}[bar % 4]) % 12\n')
        f.write(f'bass     : program {bass.program} '
                f'"{bass.name.strip() or "(unnamed)"}"\n')
        f.write(f'tempo    : {TEMPO:.0f} BPM in the written MIDI (arbitrary)\n\n')
        prog = ' | '.join(' '.join(labels[i:i + 4]) for i in range(0, n_bars, 4))
        f.write(f'progression: | {prog} |\n\n')
        f.write(f'{"bar":>3}  {"time":>6}  {"rule root":9}  {"bass struck":11}  label\n')
        for b, t0, root, struck, label in rows:
            note = (pretty_midi.note_number_to_name(struck) if struck is not None
                    else '-')
            f.write(f'{b:3d}  {t0:6.2f}  {ROOT_NAMES[root]:9}  {note:11}  {label}\n')
        f.write('\nroot is exact (what the span was selected for); quality is a '
                'guess from the other tracks.\n')
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--stage_dir', required=True)
    p.add_argument('--midi_zip', required=True)
    p.add_argument('--upstream', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--per_key', type=int, default=2,
                   help='spans per key; the longest ones are taken')
    p.add_argument('--min_bars', type=int, default=8)
    args = p.parse_args()

    spans, _, _ = stage_get(args.stage_dir)
    by_key = collections.defaultdict(list)
    for s in spans:
        if s['n_bars'] >= args.min_bars:
            by_key[s['key']].append(s)
    picked = []
    for k in range(12):
        by_key[k].sort(key=lambda s: (-s['n_bars'], s['path']))
        picked.extend(by_key[k][:args.per_key])
    print(f'{len(spans)} spans staged; writing {len(picked)} samples '
          f'({args.per_key} per key where available)')

    os.makedirs(args.out_dir, exist_ok=True)
    index = []
    with zipfile.ZipFile(args.midi_zip) as zf:
        for s in picked:
            name = write_sample(s, zf, args.upstream, args.out_dir)
            if name:
                index.append(name)
                print('  ' + name)
    with open(os.path.join(args.out_dir, 'README.txt'), 'w') as f:
        f.write('v4 listening samples. Each folder: bass.mid (training data), '
                'full.mid (same bars, all tracks),\nchords.lab / chords.txt '
                '(root per bar from the bass; quality guessed from the rest).\n\n')
        f.write('\n'.join(index) + '\n')
    print(f'\n{len(index)} samples -> {args.out_dir}')


if __name__ == '__main__':
    main()
