"""Does a generated bass dataset contain the keys and the rule you think?

Every dataset ships a <out_pt>.beat_chords.pt track: one chord token per
subbeat, written straight from the generator's own xf_chords, so the harmony
can be checked without decoding a single pitch.

Three things go wrong silently, and the tensor shape catches none of them --
beat-cycle and bar-cycle data have identical shapes:

  wrong keys        --keys omitted a key, so a split the table calls "seen"
                    was never trained on (or vice versa).
  wrong cycle_unit  --cycle_unit beat data evaluated as bar: the root changes
                    every beat instead of every bar, so a probe scoring
                    downbeats against a bar-held root reads near zero.
  wrong rule        the roots do not follow (key + OFFSETS[p]) % 12 at all.

Usage
-----
  python -m midi_adapter.verify_bass_data data/bass_finetune_bar4
  python -m midi_adapter.verify_bass_data data/bass_finetune_bar data/bass_finetune_bar4
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.chord_tokenizer import N_QUALITIES, NO_CHORD_TOKEN
from midi_adapter.generate_synthetic_bass import OFFSETS, SUBBEATS_PER_BAR

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _roots(song: torch.Tensor) -> list[int | None]:
    """Chord token per subbeat -> root semitone per subbeat (None = no chord)."""
    return [None if int(t) == NO_CHORD_TOKEN else int(t) // N_QUALITIES
            for t in song]


def _compliance(roots: list[int | None], spc: int) -> tuple[int, int]:
    """(hits, total) for roots[t] == (key + OFFSETS[(t // spc) % 4]) % 12.

    spc=1 is the per-beat rule (a cycle fills one bar), spc=4 the per-bar rule
    (a cycle spans four bars). The key is taken from subbeat 0, which every
    pattern starts on.
    """
    if not roots or roots[0] is None:
        return 0, 0
    key = roots[0]
    hits = total = 0
    for t, r in enumerate(roots):
        if r is None:
            continue
        total += 1
        hits += int(r == (key + OFFSETS[(t // spc) % 4]) % 12)
    return hits, total


def report(prefix: str) -> bool:
    prefix = prefix[:-3] if prefix.endswith('.pt') else prefix
    cp = f'{prefix}.beat_chords.pt'
    if not os.path.exists(cp):
        print(f'{prefix}: MISSING {cp}')
        return False

    songs = torch.load(cp)
    lens = torch.load(f'{prefix}.length.pt')
    n = len(songs)
    sub = int(lens[0]) if len(lens) else 0

    keys = Counter()
    bar_h = bar_t = beat_h = beat_t = 0
    changes_per_bar = Counter()
    for s in songs:
        r = _roots(s)
        if r and r[0] is not None:
            keys[r[0]] += 1
        h, t = _compliance(r, 1)                                # spc = 1
        beat_h += h; beat_t += t
        h, t = _compliance(r, 4)                                # spc = 4
        bar_h += h; bar_t += t
        # How often does the root actually move? 4 changes per bar means
        # beat-cycle data; 1 per bar means bar-cycle.
        for b in range(len(r) // SUBBEATS_PER_BAR):
            blk = r[b * SUBBEATS_PER_BAR:(b + 1) * SUBBEATS_PER_BAR]
            changes_per_bar[len({x for x in blk if x is not None})] += 1

    beat_acc = beat_h / beat_t if beat_t else float('nan')
    bar_acc = bar_h / bar_t if bar_t else float('nan')
    distinct = changes_per_bar.most_common(1)[0][0] if changes_per_bar else 0
    mode = ('bar' if distinct == 1 else
            'beat' if distinct >= 3 else f'?({distinct} roots/bar)')

    klist = sorted(keys)
    print(f'{prefix}')
    print(f'  songs={n}  subbeats/song={sub}  bars/song={sub // SUBBEATS_PER_BAR}')
    print(f'  keys  : {{{",".join(str(k) for k in klist)}}}  '
          f'({", ".join(ROOT_NAMES[k] for k in klist)})')
    print(f'  counts: ' + '  '.join(f'{ROOT_NAMES[k]}={keys[k]}' for k in klist))
    print(f'  cycle : {mode}   (distinct roots per bar = {distinct})')
    print(f'  rule  : per-bar  {bar_acc:.4f}      per-beat {beat_acc:.4f}')
    if bar_acc < 0.99 and beat_acc < 0.99:
        print('  !! neither rule fits -- this data does not follow '
              '(key + OFFSETS[p]) % 12')
    elif beat_acc > bar_acc:
        print('  !! this is --cycle_unit BEAT data. Training or scoring it '
              'with beats_per_chord=4 will read near zero.')
    print()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prefixes', nargs='+',
                   help='Dataset prefixes, with or without the .pt suffix.')
    args = p.parse_args()
    ok = all([report(x) for x in args.prefixes])
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
