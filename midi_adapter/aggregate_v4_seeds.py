"""Combine several seeds' v4_eval tables: every tier, mean ± std across seeds.

aggregate_seeds reads only the downbeat metric and v3's five key groups. The
v4/v5 tables carry the histogram tiers under 'metrics' and the two-way key
groups under 'rows'; this reads those. Run names carry a _s<seed> suffix,
which is stripped so one model lines up across seeds.

Usage
-----
  python -m midi_adapter.aggregate_v4_seeds logs_v5/table_s*_p16_T0.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st

from midi_adapter.evaluate_cp_bass import HARMONY_KEYS

FIRST = ('Data', 'Pretrain', 'Finetune')


def strip_seed(label: str) -> str:
    return re.sub(r'_s\d+$', '', label)


def fmt(v: list[float], width: int) -> str:
    v = [x for x in v if x == x]                      # drop nan (an empty key group)
    if not v:
        return f'{"--":>{width}}'
    if len(v) == 1:
        return f'{v[0]:>{width}.3f}'
    return f'{st.mean(v):>{width - 6}.3f}±{st.pstdev(v):<5.3f}'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('json_files', nargs='+')
    p.add_argument('--out', default=None, help='write the means and stds as JSON')
    args = p.parse_args()

    blobs = []
    for fp in args.json_files:
        with open(fp) as f:
            blobs.append(json.load(f))
    protocol = blobs[0]['protocol']
    rows = [r[0] for r in blobs[0]['rows']]
    metrics = list(blobs[0]['metrics'])

    # tier -> model -> row -> [value per seed]
    acc: dict[str, dict[str, dict[str, list[float]]]] = {m: {} for m in metrics}
    chord: dict[str, dict[str, dict[str, list[float]]]] = {}
    for b in blobs:
        for m in metrics:
            for label, per_row in b['metrics'].get(m, {}).items():
                d = acc[m].setdefault(strip_seed(label), {})
                for r, v in per_row.items():
                    d.setdefault(r, []).append(float(v))
        for label, per_row in b['chord_tone'].items():
            d = chord.setdefault(strip_seed(label), {})
            for r, prof in per_row.items():
                for k, v in prof.items():
                    d.setdefault(f'{r}/{k}', []).append(float(v))

    labels = list(acc[metrics[0]])
    models = [l for l in FIRST if l in labels] + [l for l in labels if l not in FIRST]
    n = len(blobs)
    w = 20

    print(f'{n} table(s): ' + ', '.join(args.json_files))
    print(f'{protocol["n_prompt_beats"]}-step prompt, {protocol["n_gen_beats"]} '
          f'generated, {protocol["n_trials"]} trials, temperature '
          f'{protocol["temperature"]}; mean ± std over seeds')
    for m in metrics:
        print()
        print(f'{m}')
        print(f'{"":<16}' + ''.join(f'{l[:18]:>{w}}' for l in models))
        for r in rows:
            print(f'{r:<16}' + ''.join(fmt(acc[m].get(l, {}).get(r, []), w) for l in models))

    print()
    print('harmonic profile, Unseen row')
    print(f'{"":<16}' + ''.join(f'{k:>{w}}' for k in HARMONY_KEYS))
    for l in models:
        print(f'{l[:16]:<16}' + ''.join(fmt(chord.get(l, {}).get(f'Unseen/{k}', []), w)
                                        for k in HARMONY_KEYS))

    found = {l: len(acc[metrics[0]][l].get(rows[0], [])) for l in labels}
    short = {l: k for l, k in found.items() if k != n}
    if short:
        print()
        print('NOT in every table:')
        for l, k in short.items():
            print(f'  {l}: {k} of {n}')

    print()
    print('seed spread on Unseen (max - min), per tier:')
    for m in metrics:
        for l in models:
            v = acc[m].get(l, {}).get('Unseen', [])
            if len(v) > 1 and l not in FIRST:
                print(f'  {m:<15} {l:<28} {min(v):.3f} .. {max(v):.3f}   spread {max(v) - min(v):.3f}')

    if args.out:
        def stats(v):
            v = [x for x in v if x == x]
            return {'mean': st.mean(v) if v else float('nan'),
                    'std': st.pstdev(v) if v else float('nan'), 'n': len(v)}
        out = {'protocol': protocol, 'n_seeds': n, 'files': args.json_files,
               'metrics': {m: {l: {r: stats(v) for r, v in acc[m][l].items()}
                               for l in acc[m]} for m in metrics}}
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
