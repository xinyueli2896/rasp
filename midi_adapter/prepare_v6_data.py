"""
Build the v6 dataset: spans whose CHORD cadence matches one of the 24.

v4 and v5 filtered on roots, so their spans pool the major and minor cadence
on each tonic -- the two have identical root sequences. v6's rule is over
chords (models/chord_tracr_rule_model.py), so this filter asks for the whole
cadence:

    tonic chord = (root of bar 0's downbeat bass, quality of bar 0's third)
    span kept    if every bar's chord matches CADENCE[tonic chord]

Reads a stage written by prepare_v4_data -- the chordal one, since a bass line
alone voices no third and so carries no chord quality to read. No MIDI, no
rescan, seconds to run.

    python -m midi_adapter.prepare_v6_data \\
        --stage $WORK/stage_chordal --out_prefix $WORK/data/v6_chordal \\
        --hold_out F# D#m G# Fm

The split is TWO-way: the held-out tonic chords, and everything else. v4's
three-way finetune/unseen/pretrain split does not carry over -- it keys off
the tonic alone (PRETRAIN_ONLY = [5, 9]), which cannot express "F minor is
held out but F major is not", and v4's own driver already noted the pretrain
row is vestigial since the frozen host was pretrained on the whole LA corpus.

Why holding out a key rather than a tonic is sound here: under chord symbols a
tonic chord's parallel partner shares exactly ONE of its four chords (the V,
major in both modes), where under root symbols it shared all four roots. And
each held-out tonic chord still appears in trained cadences at NON-tonic
positions -- F# major as the IV of C# major and the V of B major and B minor --
exactly as v1-v5's held-out starters appeared as non-initial tokens.

Outputs, per split, in the layout FramedDataset reads:
    <prefix>_<split>.pt / .length.pt / .pitch_shift_range.pt / .txt
    <prefix>_<split>.chords.pt   tonic chord index per song, 0..23
    <prefix>_<split>.keys.pt     its root, for tooling that expects v4's name
    <prefix>.manifest.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

from midi_adapter.prepare_v4_data import (
    OFFSETS, BAR_SUBBEATS, CYCLE_BARS, MAX_POLYPHONY, stage_get, stage_done,
    transpose,
)
from midi_adapter.census_v6_quality import bar_quality as _bar_quality
from midi_adapter.census_v6_quality import MAJ, MIN, AMB, NONE
from models.chord_tracr_rule_model import (
    ChordTracrRuleModel, CHORD_NAMES, N_CHORDS, chord_index, chord_root,
    chord_is_minor,
)

QUAL_OF = {MAJ: False, MIN: True}          # symbol -> is_minor


def bar_quality(*a, **k) -> str:
    """Just the symbol. census_v6_quality.bar_quality also returns the two
    masses it compared, which only its own reporting needs."""
    return _bar_quality(*a, **k)[0]


# ---------------------------------------------------------------------------
# One span
# ---------------------------------------------------------------------------

def span_chords(song: dict, rule: ChordTracrRuleModel, *, bar_len: int,
                max_polyphony: int, exclude_bass: bool, margin: float,
                allow_unvoiced: bool):
    """(tonic_chord, per-bar quality symbols) or (None, reason, quals).

    The root sequence is already guaranteed by prepare_v4_data's rule filter,
    so what is checked here is the quality of each bar against the cadence the
    tonic chord implies.
    """
    data = song['data'].numpy()
    n_bars = len(data) // bar_len
    # The rule root of each bar is the span's own key plus the phase offset,
    # as prepare_v4_data established when it selected the span.
    quals = [bar_quality(data[b * bar_len:(b + 1) * bar_len],
                         (song['key'] + OFFSETS[b % CYCLE_BARS]) % 12,
                         max_polyphony, bar_len, exclude_bass, margin)
             for b in range(n_bars)]

    if quals[0] not in QUAL_OF:
        return None, 'bar0_unreadable', quals
    tonic = chord_index(song['key'], QUAL_OF[quals[0]])
    want_minor = [chord_is_minor(int(c)) for c in rule.cadence[tonic]]

    for b, q in enumerate(quals):
        exp = MIN if want_minor[b % CYCLE_BARS] else MAJ
        if q == exp:
            continue
        if q in (NONE, AMB):
            if allow_unvoiced:
                continue
            return None, 'bar_unvoiced', quals
        return None, 'cadence_mismatch', quals
    return tonic, None, quals


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_split(prefix: str, split: str, songs: list[dict]) -> int:
    if not songs:
        print(f'  {split:9s}: empty, nothing written')
        return 0
    name = f'{prefix}_{split}'
    data = torch.cat([s['data'] for s in songs], dim=0)
    lengths = torch.tensor([len(s['data']) for s in songs])
    torch.save(data, f'{name}.pt')
    torch.save(lengths, f'{name}.length.pt')
    # [0, 0] per song: FramedDataset samples from this range, and a transposed
    # window is a different tonic CHORD, which is what the split partitions on.
    torch.save(torch.zeros(len(songs), 2, dtype=torch.int8).reshape(-1),
               f'{name}.pitch_shift_range.pt')
    torch.save(torch.tensor([s['tonic'] for s in songs], dtype=torch.int16),
               f'{name}.chords.pt')
    torch.save(torch.tensor([chord_root(s['tonic']) for s in songs],
                            dtype=torch.int8), f'{name}.keys.pt')
    with open(f'{name}.txt', 'w') as f:
        for i, s in enumerate(songs):
            shift = f'+{s["shift"]}' if s.get('shift') else ''
            f.write(f'{i}\t{os.path.basename(s["path"])}@bar{s["start_bar"]}'
                    f'{shift}\tchord={CHORD_NAMES[s["tonic"]]}\t'
                    f'bars={s["n_bars"]}\n')
    print(f'  {split:9s}: {len(songs):6d} songs, {int(lengths.sum()):9d} subbeats, '
          f'{tuple(data.shape)}')
    return len(songs)


def check_invariants(songs: list[dict], rule: ChordTracrRuleModel,
                     bar_len: int, max_polyphony: int, *, exclude_bass: bool,
                     margin: float) -> bool:
    """The three things a v6 run depends on. All must read 1.000.

    The first is new to v6: the rule model is handed a tonic CHORD, so the
    prompt has to disclose both halves of it -- the root from the downbeat
    bass in slot 0, the quality from bar 0's third. A misread tonic chord
    makes the rule model compute a consistent but DIFFERENT cadence.
    """
    tonic_ok = bar_ok = bar_n = cycle_ok = 0
    for s in songs:
        d = s['data'].view(-1, max_polyphony, 4)
        data = s['data'].numpy()
        root = chord_root(s['tonic'])
        q0 = bar_quality(data[:bar_len], root, max_polyphony, bar_len,
                         exclude_bass, margin)
        slot0 = int(d[0, 0, 1])
        if (slot0 != 255 and int(d[0, 0, 0]) < 127 and slot0 % 12 == root
                and q0 in QUAL_OF and QUAL_OF[q0] == chord_is_minor(s['tonic'])):
            tonic_ok += 1
        want = rule.cadence[s['tonic']]
        for b in range(len(d) // bar_len):
            bar_n += 1
            exp = int(want[b % CYCLE_BARS])
            q = bar_quality(data[b * bar_len:(b + 1) * bar_len],
                            chord_root(exp), max_polyphony, bar_len,
                            exclude_bass, margin)
            step = d[b * bar_len]
            if (int(step[0, 1]) % 12 == chord_root(exp)
                    and q in QUAL_OF and QUAL_OF[q] == chord_is_minor(exp)):
                bar_ok += 1
        if len(s['data']) % (bar_len * CYCLE_BARS) == 0:
            cycle_ok += 1
    n = max(len(songs), 1)
    print('\ninvariants (all three must be 1.000):')
    print(f'  tonic CHORD readable from bar 0 : {tonic_ok / n:.3f}  '
          f'({tonic_ok}/{len(songs)} songs)')
    print(f'  every bar matches the cadence    : {bar_ok / max(bar_n, 1):.3f}  '
          f'({bar_ok}/{bar_n} bars)')
    print(f'  length is a whole cycle          : {cycle_ok / n:.3f}  '
          f'({cycle_ok}/{len(songs)} songs)')
    return tonic_ok == len(songs) and bar_ok == bar_n and cycle_ok == len(songs)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--stage', required=True, help='the chordal stage')
    p.add_argument('--out_prefix', required=True)
    p.add_argument('--hold_out', nargs='*', default=['F#', 'D#m', 'G#', 'Fm'],
                   help='tonic CHORDS held out of finetuning, by name '
                        '(F# is F# major, F#m is F# minor)')
    p.add_argument('--min_bars', type=int, default=8)
    p.add_argument('--bar_subbeats', type=int, default=BAR_SUBBEATS)
    p.add_argument('--max_polyphony', type=int, default=MAX_POLYPHONY)
    p.add_argument('--minor_pattern', default='harmonic',
                   choices=['harmonic', 'natural'])
    p.add_argument('--margin', type=float, default=0.25)
    p.add_argument('--include_bass', action='store_true',
                   help='let slot 0 vote on the harmony of its bar')
    p.add_argument('--allow_unvoiced', action='store_true',
                   help='keep a span whose bars never CONTRADICT the cadence '
                        'even where a bar voices no third. More data, but the '
                        'model then sees bars with no third in them')
    p.add_argument('--balance', type=int, default=0,
                   help='transpose spans so each NON-held-out tonic chord has '
                        'at least this many. Never transposes into a held-out '
                        'chord, which would leak it into finetuning')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--json_out', default=None)
    args = p.parse_args()

    bad = [h for h in args.hold_out if h not in CHORD_NAMES]
    if bad:
        sys.exit(f'unknown chord name(s) {bad}; expected e.g. F# or F#m')
    held = {CHORD_NAMES.index(h) for h in args.hold_out}
    rule = ChordTracrRuleModel(minor_pattern=args.minor_pattern)

    songs, _, _ = stage_get(args.stage)
    if not songs:
        sys.exit(f'{args.stage} holds no spans')
    songs = [s for s in songs if s['n_bars'] >= args.min_bars]
    print(f'{len(songs)} staged spans of >= {args.min_bars} bars, from '
          f'{len(stage_done(args.stage))} files scanned')
    print(f'rule: {args.minor_pattern} minor; held out '
          f'{" ".join(CHORD_NAMES[c] for c in sorted(held))}')

    kept: list[dict] = []
    why = collections.Counter()
    for s in songs:
        tonic, reason, _ = span_chords(
            s, rule, bar_len=args.bar_subbeats,
            max_polyphony=args.max_polyphony,
            exclude_bass=not args.include_bass, margin=args.margin,
            allow_unvoiced=args.allow_unvoiced)
        if tonic is None:
            why[reason] += 1
            continue
        kept.append({**s, 'tonic': tonic, 'shift': 0})
        why['kept'] += 1

    print(f'\nspans by outcome:')
    for k, v in why.most_common():
        print(f'  {k:20s} {v:6d}  {v / max(len(songs), 1):.3f}')
    if not kept:
        sys.exit('\nnothing matched a cadence. If bar0_unreadable dominates, '
                 'the chord tracks do not voice a third on the first bar; if '
                 'cadence_mismatch does, these spans follow the root rule but '
                 'not one of the 24 chord cadences.')

    by_chord = collections.defaultdict(list)
    for s in kept:
        by_chord[s['tonic']].append(s)

    if args.balance:
        rng = np.random.default_rng(args.seed)
        donors = sorted(by_chord, key=lambda c: -len(by_chord[c]))
        added = collections.Counter()
        for c in range(N_CHORDS):
            if c in held:
                continue          # never fill a held-out chord
            pool = [d for d in donors
                    if d not in held and chord_is_minor(d) == chord_is_minor(c)
                    and len(by_chord[d]) > args.balance]
            while len(by_chord[c]) < args.balance and pool:
                d = pool[int(rng.integers(len(pool)))]
                src = by_chord[d][int(rng.integers(len(by_chord[d])))]
                if src.get('shift'):
                    continue
                semi = (chord_root(c) - chord_root(d)) % 12
                semi = semi - 12 if semi > 6 else semi
                moved = transpose(src['data'], semi, args.max_polyphony)
                if moved is None:
                    pool.remove(d)
                    continue
                by_chord[c].append({**src, 'data': moved, 'tonic': c,
                                    'shift': semi})
                added[c] += 1
        print(f'\ntransposed in to reach {args.balance} per chord: '
              f'{ {CHORD_NAMES[c]: n for c, n in sorted(added.items())} }')
        print('  (quality is unchanged by transposition, so a span only ever '
              'moves within its own mode, and never into a held-out chord)')

    print('\nper tonic chord:')
    for mino in (False, True):
        row = []
        for r in range(12):
            c = chord_index(r, mino)
            tag = '*' if c in held else ' '
            row.append(f'{CHORD_NAMES[c]:>4}{tag}{len(by_chord[c]):>5}')
        print('  ' + '  '.join(row[:6]))
        print('  ' + '  '.join(row[6:]))
    print('  * held out')

    splits = {'unseen': sorted(held),
              'finetune': [c for c in range(N_CHORDS) if c not in held],
              'all': list(range(N_CHORDS))}
    all_songs = [s for c in range(N_CHORDS) for s in by_chord[c]]
    ok = check_invariants(all_songs, rule, args.bar_subbeats,
                          args.max_polyphony,
                          exclude_bass=not args.include_bass,
                          margin=args.margin)
    if not ok:
        print('  !! an invariant failed -- do not train on this; the rule '
              'model would be handed a tonic chord the data does not state')

    print('\nwriting:')
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    counts = {}
    for split, chords in splits.items():
        sel = [s for c in chords for s in by_chord[c]]
        sel.sort(key=lambda s: (s['tonic'], s['path'], s['start_bar']))
        counts[split] = write_split(args.out_prefix, split, sel)

    manifest = {
        'rule': 'chord cadence', 'minor_pattern': args.minor_pattern,
        'vocab': N_CHORDS, 'bar_subbeats': args.bar_subbeats,
        'cycle_subbeats': args.bar_subbeats * CYCLE_BARS,
        'min_bars': args.min_bars, 'balance': args.balance,
        'allow_unvoiced': args.allow_unvoiced,
        'held_out': [CHORD_NAMES[c] for c in sorted(held)],
        'splits': {k: {'chords': [CHORD_NAMES[c] for c in v],
                       'songs': counts[k]} for k, v in splits.items()},
        'outcomes': dict(why),
        'per_chord': {CHORD_NAMES[c]: len(by_chord[c]) for c in range(N_CHORDS)},
        'invariants_ok': bool(ok),
        'songs': [{k: v for k, v in s.items() if k != 'data'}
                  for s in all_songs],
    }
    path = args.json_out or f'{args.out_prefix}.manifest.json'
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'\nmanifest -> {path}')
    print(f'train with --beats_per_chord {args.bar_subbeats} '
          f'(cycle = {args.bar_subbeats * CYCLE_BARS} subbeats) and a rule '
          f'model of d_model {rule.d_model}')


if __name__ == '__main__':
    main()
