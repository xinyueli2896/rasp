"""Combine several seeds' result tables into one mean ± std table.

Each seed's run names carry a _s<seed> suffix, so the suffix is stripped to
line the same model up across seeds. A model missing from a seed is reported
with the count it was actually found in rather than being silently averaged
over fewer runs.

Usage
-----
  python -m midi_adapter.aggregate_seeds logs_bar/table_s*.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys


ROW_ORDER = ['Pretrain-only', 'Finetune-only', 'Both seen', 'Unseen', 'Overall']


def strip_seed(label: str) -> str:
    """cp_yy_frozen_s2 -> cp_yy_frozen, so seeds line up in one column."""
    return re.sub(r'_s\d+$', '', label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('json_files', nargs='+')
    p.add_argument('--sort_by', default='Unseen',
                   help='Row to order the columns by (default Unseen). '
                        'Baselines keep their place at the front.')
    args = p.parse_args()

    # model -> row -> [value per seed]
    acc: dict[str, dict[str, list[float]]] = {}
    protocol = None
    for fp in args.json_files:
        with open(fp) as f:
            blob = json.load(f)
        protocol = protocol or blob.get('protocol')
        for label, rows in blob['results'].items():
            m = acc.setdefault(strip_seed(label), {})
            for row, v in rows.items():
                m.setdefault(row, []).append(float(v))

    n_seeds = len(args.json_files)
    baselines = [m for m in ('Pretrain', 'Finetune') if m in acc]
    others = [m for m in acc if m not in baselines]
    others.sort(key=lambda m: -st.mean(acc[m].get(args.sort_by, [0])))
    models = baselines + others

    rows = [r for r in ROW_ORDER if any(r in acc[m] for m in models)]
    w = {m: max(len(m), 13) for m in models}

    if protocol:
        print(f"protocol: {protocol['n_prompt_beats']}-beat prompt, "
              f"{protocol['n_gen_beats']} generated, "
              f"{protocol['n_trials']} trials, "
              f"beats_per_chord={protocol['beats_per_chord']}")
    print(f'{n_seeds} seeds: ' + ', '.join(args.json_files))
    print()

    header = f'{"Key group":<16}' + ''.join(f' {m:>{w[m]}}' for m in models)
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for r in rows:
        if r == 'Overall':
            print('-' * len(header))
        line = f'{r:<16}'
        for m in models:
            v = acc[m].get(r, [])
            if not v:
                line += f' {"--":>{w[m]}}'
            elif len(v) == 1:
                line += f' {v[0]:>{w[m]}.3f}'
            else:
                line += f' {st.mean(v):>{w[m]-6}.3f}±{st.pstdev(v):<5.3f}'
        print(line)
    print('=' * len(header))

    short = {m: len(v.get('Overall', [])) for m, v in acc.items()}
    missing = {m: n for m, n in short.items() if n != n_seeds}
    if missing:
        print()
        print('NOT averaged over all seeds:')
        for m, n in sorted(missing.items()):
            print(f'  {m}: found in {n} of {n_seeds}')

    # Spread matters as much as the mean here: if a model's seed range is
    # wider than the gap between models, no ordering between them is real.
    print()
    print(f'seed spread on {args.sort_by} (max - min):')
    for m in models:
        v = acc[m].get(args.sort_by, [])
        if len(v) > 1:
            print(f'  {m:<28} {min(v):.3f} .. {max(v):.3f}   '
                  f'spread {max(v) - min(v):.3f}')


if __name__ == '__main__':
    main()
