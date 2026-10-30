"""
Union several v4_eval tables into one.

Two nodes can each evaluate their own stems (run_v4_stems.sh with EVAL_STEMS
a subset writes table_..._<stems>.json); this puts the columns side by side
and prints the result exactly as v4_eval would have. Every input must share
the protocol -- prompt length, generated length, trials, temperature and
prompt files -- or the columns are not comparable, and the merge refuses.

When two inputs carry the same model, the LATER file wins. That is the
intended order: the full-stem table a node wrote while the other node's
models were still training may hold a half-trained column for them, and the
other node's own table, listed second, replaces it.

  python -m midi_adapter.merge_tables logs_v4/table_s1_p16_T0.json \\
      logs_v4/table_s1_p16_T0_trainar-scratch.json \\
      --out logs_v4/table_s1_p16_T0_merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.v4_eval import print_tables, METRICS
from midi_adapter.evaluate_cp_bass import UNSEEN

PROTOCOL_KEYS = ('n_prompt_beats', 'n_gen_beats', 'n_trials', 'beats_per_chord',
                 'temperature', 'prompt_data', 'prompt_split')
# Column order of the merged table: reference rows first, then adapters in
# the order they were first seen.
FIRST = ('Data', 'Pretrain', 'Finetune')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('json_files', nargs='+')
    p.add_argument('--out', default=None, help='write the merged JSON here')
    args = p.parse_args()

    blobs = []
    for f in args.json_files:
        with open(f) as fh:
            blobs.append((f, json.load(fh)))

    proto = {k: blobs[0][1]['protocol'].get(k) for k in PROTOCOL_KEYS}
    for f, b in blobs[1:]:
        other = {k: b['protocol'].get(k) for k in PROTOCOL_KEYS}
        if other != proto:
            diff = {k: (proto[k], other[k]) for k in PROTOCOL_KEYS if proto[k] != other[k]}
            sys.exit(f'{f}: protocol differs from {blobs[0][0]}: {diff}')
    rows = next((b.get('rows') for _, b in blobs if b.get('rows')), None)
    if rows is None:
        # A table written before 'rows' was recorded: the v4 two-way groups.
        seen = [k for k in range(12) if k not in UNSEEN]
        rows = [('Finetuned keys', seen, '{' + ','.join(map(str, seen)) + '}'),
                ('Unseen', UNSEEN, '{' + ','.join(map(str, UNSEEN)) + '}'),
                ('Overall', list(range(12)), 'all 12 keys')]
    rows = [tuple(r) for r in rows]

    results, metrics, chord, per_key = {}, {m: {} for m in METRICS}, {}, {}
    origin = {}
    for f, b in blobs:
        for lbl in b['results']:
            if lbl in results:
                print(f'  {lbl:<28} in {origin[lbl]} and {f}: keeping {f}')
            origin[lbl] = f
            results[lbl] = b['results'][lbl]
            for m in METRICS:
                metrics[m][lbl] = b['metrics'][m][lbl]
            chord[lbl] = b['chord_tone'][lbl]
            per_key[lbl] = b['per_key'][lbl]

    order = [l for l in FIRST if l in results] + [l for l in results if l not in FIRST]
    reorder = lambda d: {l: d[l] for l in order}
    results, chord, per_key = reorder(results), reorder(chord), reorder(per_key)
    metrics = {m: reorder(metrics[m]) for m in METRICS}

    print(f'merged {len(blobs)} tables, {len(order)} columns: {", ".join(order)}')
    print_tables(blobs[0][1]['protocol'], rows, results, metrics, chord, per_key)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump({'protocol': blobs[0][1]['protocol'], 'results': results,
                       'metrics': metrics, 'chord_tone': chord,
                       'rows': [list(r) for r in rows], 'per_key': per_key,
                       'merged_from': {l: origin[l] for l in order}}, fh, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
