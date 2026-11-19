"""
v6 census: what chord QUALITIES does the rule-following data actually carry?

OFFSETS = [0, 5, 7, 0] are ROOT intervals, and they are the same for major
I-IV-V-I and minor i-iv-v-i -- only the thirds differ. So the v4/v5 filter,
which tests roots alone, has been collecting both all along, and the quality is
exactly the information the rule channel drops. v6 adds it back.

Which means the v6 rule cannot be chosen from the armchair. It is

    root[b] = (key + OFFSETS[b % 4]) % 12          (unchanged)
    qual[b] = QUAL[mode][b % 4]

and QUAL has to match what the corpus plays. Three candidates:

    major            M M M M     constant: the quality channel says nothing
    natural minor    m m m m     one bit per song: "copy the mode"
    harmonic minor   m m M m     the third FLIPS on the V chord only

Only the third makes quality vary inside the cycle, which is what forces a
model to use the rule rather than track roots. Whether real minor-key music in
this corpus writes a major V or a modal v decides whether v6 is a strong
experiment or a trivial one, so this script counts before anything is built.

It reads a stage written by prepare_v4_data -- the span tensors, not the
corpus -- so it needs no MIDI, no download and no rescan, and runs in seconds.

    python -m midi_adapter.census_v6_quality --stage $WORK/stage_chordal

What it reports
---------------
  * the 4-bar quality pattern of every cycle, as a histogram
  * how many spans hold ONE pattern throughout (a span that wanders is
    describable by no QUAL[mode] and has to be dropped)
  * how many spans have bar 0's third readable -- v6's new invariant, since
    the rule model must read the mode from the prompt the way it already reads
    the key, and one note cannot carry a mode
  * counts per (key, mode) cell, which is what decides whether the held-out
    set can become (key, mode) pairs instead of keys
  * how many spans survive each candidate rule

Quality is read as the third and nothing else: it is the one interval always
voiced, where sevenths and ninths are optional colour, and it matches the
root/3rd/5th terms the v4/v5 metrics already use.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

from midi_adapter.prepare_v4_data import (
    OFFSETS, ROOT_NAMES, BAR_SUBBEATS, CYCLE_BARS, UNSEEN, stage_get,
    stage_done,
)
from midi_adapter.generate_synthetic_bass import DURATION_TEMPLATES

MAJ, MIN, AMB, NONE = 'M', 'm', '?', '.'

# The candidate rules, as the module docstring sets them out. A span matches a
# candidate when every one of its cycles shows that candidate's pattern.
CANDIDATES = {
    'major':          (MAJ, MAJ, MAJ, MAJ),
    'natural minor':  (MIN, MIN, MIN, MIN),
    'harmonic minor': (MIN, MIN, MAJ, MIN),
}


def bar_quality(step_rows: np.ndarray, root: int, max_polyphony: int,
                bar_len: int, exclude_bass: bool, margin: float):
    """Quality of one bar: mass on the major third against the minor third.

    step_rows is the bar's (bar_len, max_polyphony * 4) slice of raw CP, whose
    tuples are (program, pitch, duration_index, velocity) with 255 marking pad
    and program 254 the EOS slot. Weight is the note's duration CLIPPED AT THE
    BAR LINE, which is what v4_eval's scorer does -- a chord held across the
    barline should not vote in the next bar in one place and not the other.

    exclude_bass drops slot 0, which v5 guarantees holds the bass: a bass
    passing tone through the third would otherwise vote on the harmony.
    """
    d = step_rows.reshape(bar_len, max_polyphony, 4).astype(np.int32)
    prog, pitch, dur = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    valid = (pitch != 255) & (prog != 254) & (prog != 127)
    if exclude_bass:
        valid[:, 0] = False
    t = np.arange(bar_len)[:, None]
    length = DURATION_TEMPLATES[np.clip(dur, 0, len(DURATION_TEMPLATES) - 1)]
    weight = np.clip(np.minimum(length, bar_len - t), 0, None) * valid
    rel = (pitch - root) % 12
    maj = float(weight[(rel == 4) & valid].sum())
    mina = float(weight[(rel == 3) & valid].sum())
    if maj <= 0 and mina <= 0:
        return NONE, maj, mina
    if maj > 0 and mina > 0 and min(maj, mina) > margin * max(maj, mina):
        return AMB, maj, mina
    return (MAJ if maj > mina else MIN), maj, mina


def span_patterns(song: dict, max_polyphony: int, bar_len: int,
                  exclude_bass: bool, margin: float):
    """Per-bar qualities and per-cycle 4-bar patterns of one staged span."""
    data = song['data'].numpy()
    n_bars = len(data) // bar_len
    quals = []
    for b in range(n_bars):
        root = (song['key'] + OFFSETS[b % CYCLE_BARS]) % 12
        q, _, _ = bar_quality(data[b * bar_len:(b + 1) * bar_len], root,
                              max_polyphony, bar_len, exclude_bass, margin)
        quals.append(q)
    cycles = [tuple(quals[i:i + CYCLE_BARS])
              for i in range(0, n_bars - CYCLE_BARS + 1, CYCLE_BARS)]
    return quals, cycles


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--stage', required=True,
                   help='a stage directory written by prepare_v4_data, '
                        'normally the chordal (v5) one -- bass-only spans '
                        'voice no third and cannot carry a quality')
    p.add_argument('--max_polyphony', type=int, default=16)
    p.add_argument('--bar_subbeats', type=int, default=BAR_SUBBEATS)
    p.add_argument('--min_bars', type=int, default=0,
                   help='ignore staged spans shorter than this, to census the '
                        'same spans a dataset built at that floor would hold')
    p.add_argument('--include_bass', action='store_true',
                   help='let slot 0 (the bass) vote on the harmony too. Off by '
                        'default: a bass passing tone through the third would '
                        'decide a bar')
    p.add_argument('--margin', type=float, default=0.25,
                   help='a bar counts as ambiguous when the weaker third holds '
                        'more than this fraction of the stronger one')
    p.add_argument('--json_out', default=None)
    args = p.parse_args()

    songs, _, _ = stage_get(args.stage)
    if not songs:
        sys.exit(f'{args.stage} holds no spans. If the chordal pass logged '
                 "'no_chordal_track' for everything, these songs carry no "
                 'track that both sits in a chordal GM family and plays 2+ '
                 'notes on --chord_frac of its steps, so no quality can be '
                 'read and v6 has no data. Loosen --chord_frac and re-run the '
                 'cheap pass.')
    if args.min_bars:
        songs = [s for s in songs if s['n_bars'] >= args.min_bars]
    print(f'{len(songs)} spans from {len(stage_done(args.stage))} files '
          f'scanned, {sum(s["n_bars"] for s in songs)} bars')
    print(f'third read from {"every note" if args.include_bass else "the non-bass slots"}'
          f', ambiguity margin {args.margin}')

    bar_hist = collections.Counter()
    cycle_hist = collections.Counter()
    consistent = collections.Counter()      # pattern -> spans holding it throughout
    mixed = 0
    bar0 = collections.Counter()
    cells = collections.Counter()           # (key, pattern) -> spans
    per_song = []

    for s in songs:
        quals, cycles = span_patterns(s, args.max_polyphony, args.bar_subbeats,
                                      not args.include_bass, args.margin)
        bar_hist.update(quals)
        cycle_hist.update(cycles)
        bar0[quals[0] if quals else NONE] += 1
        uniq = set(cycles)
        if len(uniq) == 1:
            pat = cycles[0]
            consistent[pat] += 1
            cells[(s['key'], pat)] += 1
        else:
            mixed += 1
            pat = None
        per_song.append({'path': s['path'], 'key': s['key'],
                         'n_bars': s['n_bars'],
                         'quals': ''.join(quals),
                         'pattern': ''.join(pat) if pat else None})

    tot_bars = sum(bar_hist.values())
    print('\nbars by what their third says:')
    for k, label in ((MAJ, 'major third'), (MIN, 'minor third'),
                     (AMB, 'both (ambiguous)'), (NONE, 'neither voiced')):
        print(f'  {label:20s} {bar_hist[k]:8d}  {bar_hist[k] / max(tot_bars, 1):.3f}')

    print('\n4-bar quality patterns, by cycle:')
    for pat, n in cycle_hist.most_common(12):
        named = next((k for k, v in CANDIDATES.items() if v == pat), '')
        print(f'  {"".join(pat):6s} {n:8d}  {n / max(sum(cycle_hist.values()), 1):.3f}'
              f'   {named}')

    print(f'\nspans holding ONE pattern throughout: '
          f'{sum(consistent.values())}/{len(songs)}; mixed: {mixed}')
    for pat, n in consistent.most_common(8):
        named = next((k for k, v in CANDIDATES.items() if v == pat), '')
        print(f'  {"".join(pat):6s} {n:6d} spans   {named}')

    print('\nbar 0, the one the mode must be read from (v6 invariant):')
    for k, label in ((MAJ, 'major third'), (MIN, 'minor third'),
                     (AMB, 'both -- unreadable'), (NONE, 'neither -- unreadable')):
        print(f'  {label:22s} {bar0[k]:6d}  {bar0[k] / max(len(songs), 1):.3f}')
    readable = bar0[MAJ] + bar0[MIN]
    print(f'  readable                {readable:6d}  '
          f'{readable / max(len(songs), 1):.3f}')
    if readable < len(songs):
        print('  spans whose bar 0 is unreadable must be dropped: the rule '
              'model reads the mode from the prompt, and a misread mode makes '
              'it compute a consistent but DIFFERENT rule')

    print('\nsurvival under each candidate rule '
          '(spans holding that pattern throughout AND with bar 0 readable):')
    for name, pat in CANDIDATES.items():
        n = sum(1 for s, rec in zip(songs, per_song)
                if rec['pattern'] == ''.join(pat) and rec['quals'][:1] in (MAJ, MIN))
        print(f'  {name:16s} {"".join(pat):6s} {n:6d} spans')
    strong = sum(1 for rec in per_song
                 if rec['pattern'] in (''.join(CANDIDATES['major']),
                                       ''.join(CANDIDATES['harmonic minor']))
                 and rec['quals'][:1] in (MAJ, MIN))
    print(f'  major + harmonic minor  {strong:6d} spans '
          '<- the pair that makes quality vary inside the cycle')

    print('\nper (key, pattern) cell, for spans with one pattern:')
    pats = [p for p, _ in consistent.most_common(4)]
    head = f'  {"key":<10}' + ''.join(f'{"".join(p):>10}' for p in pats)
    print(head)
    for k in range(12):
        tag = ' *' if k in UNSEEN else '  '
        row = f'  {ROOT_NAMES[k]:<2s}{tag}{"":<6}' + ''.join(
            f'{cells[(k, p)]:>10}' for p in pats)
        print(row)
    print('  * currently held out. With a mode bit the held-out set can become '
          '(key, mode) CELLS instead -- train F# major and C minor, test F# '
          'minor -- which tests generalisation over two factors at once. The '
          'counts above say whether that is affordable.')

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump({'stage': args.stage,
                       'offsets': OFFSETS,
                       'margin': args.margin,
                       'include_bass': args.include_bass,
                       'bars': {k: bar_hist[k] for k in (MAJ, MIN, AMB, NONE)},
                       'cycles': {''.join(k): v for k, v in cycle_hist.items()},
                       'consistent': {''.join(k): v for k, v in consistent.items()},
                       'mixed': mixed,
                       'bar0': {k: bar0[k] for k in (MAJ, MIN, AMB, NONE)},
                       'cells': {f'{k}|{"".join(p)}': v
                                 for (k, p), v in cells.items()},
                       'songs': per_song}, f, indent=1)
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
