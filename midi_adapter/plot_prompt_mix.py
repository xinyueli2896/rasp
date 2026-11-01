"""
Prompt-mix experiment: does how the prompt bar states its chord predict how
well the continuation follows the rule?

v4_eval --per_prompt_csv writes one row per (prompt, model) with the prompt
bar's note mix -- the share of its notes on the rule root, on the third or
fifth, and on non-chord tones -- next to the unaveraged scores of the bars
generated from it. This script draws, for each outcome metric, one panel per
model with three point clouds:

    green   (share of root notes,          outcome)
    amber   (share of third/fifth notes,   outcome)
    red     (share of non-chord tones,     outcome)

plus the binned mean of each cloud, and prints Spearman correlations.
Several CSVs (one per seed) are pooled.

    python -m midi_adapter.plot_prompt_mix logs_v5/prompt_mix_s*_p16_T0.csv \\
        --out_dir figs_v5

    --weight count|dur|bass   which share: by note count (default), by
                              duration, or counting only the bass line
    --group all|finetuned|unseen   restrict to a key group
    --models Data Finetune cp_yy_lora ...   subset / order of panels
    --pool                    every model in one panel per metric
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np

CLASSES = (('root', 'root', '#2a9d3f'),
           ('chord', '3rd / 5th', '#e0a800'),
           ('nct', 'non-chord tone', '#d7301f'))
PREFIX = {'count': 'p_', 'dur': 'pdur_', 'bass': 'pbass_'}
METRIC_TITLES = {'acc': 'downbeat accuracy',
                 'root_exact': 'root_exact',
                 'root_chordtone': 'root_chordtone',
                 'bass_exact': 'bass_exact',
                 'bass_chordtone': 'bass_chordtone'}


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        m = re.search(r'_s(\d+)_', os.path.basename(p))
        seed = int(m.group(1)) if m else -1
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                r = {k: (float(v) if k not in ('model', 'key_name', 'group', 'file')
                         else v) for k, v in r.items()}
                r['seed'] = seed
                # cp_yy_lora_v5_s1 and _s2 are one model across seeds
                r['model'] = re.sub(r'_s\d+$', '', r['model'])
                rows.append(r)
    return rows


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), as scipy.stats.rankdata."""
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), float)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_n: int = 5):
    """Mean y per x bin, bins with fewer than min_n points dropped."""
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    cx, cy, cn = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= min_n:
            cx.append(x[sel].mean()); cy.append(y[sel].mean()); cn.append(int(sel.sum()))
    return np.array(cx), np.array(cy), cn


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--out_dir', default='figs')
    ap.add_argument('--metrics', nargs='+', default=['acc', 'root_exact', 'root_chordtone'])
    ap.add_argument('--weight', default='count', choices=sorted(PREFIX))
    ap.add_argument('--group', default='all', choices=['all', 'finetuned', 'unseen'])
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--pool', action='store_true')
    ap.add_argument('--bins', type=int, default=10, help='equal-width bins over 0..100%%')
    ap.add_argument('--jitter', type=float, default=0.015,
                    help='vertical jitter so the discrete scores do not stack')
    ap.add_argument('--tag', default='', help='suffix for the figure file names')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = load(args.csv)
    if args.group != 'all':
        rows = [r for r in rows if r['group'] == args.group]
    order = []
    for r in rows:
        if r['model'] not in order:
            order.append(r['model'])
    models = args.models or order
    if args.pool:
        models = ['all models']
        for r in rows:
            r['model'] = 'all models'
    pre = PREFIX[args.weight]
    seeds = sorted({r['seed'] for r in rows})
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    edges = np.linspace(0, 100, args.bins + 1)

    print(f'{len(rows)} rows, seeds {seeds}, group {args.group}, share by {args.weight}')
    print('Spearman rho between the prompt share and the outcome:')
    head = f'{"model":<26}{"metric":<16}{"n":>5}' + ''.join(f'{lbl:>18}' for _, lbl, _ in CLASSES)
    print(head)

    for metric in args.metrics:
        n_panels = len(models)
        ncols = min(4, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                                 squeeze=False, sharex=True, sharey=True)
        for ax in axes.flat[n_panels:]:
            ax.axis('off')
        for ax, model in zip(axes.flat, models):
            sub = [r for r in rows if r['model'] == model]
            y = np.array([r[metric] for r in sub])
            line = f'{model:<26}{metric:<16}{len(sub):>5}'
            for cls, lbl, color in CLASSES:
                x = 100 * np.array([r[pre + cls] for r in sub])
                yj = y + rng.uniform(-args.jitter, args.jitter, len(y))
                ax.scatter(x, yj, s=9, alpha=0.35, color=color, edgecolors='none',
                           label=lbl)
                bx, by, _ = binned(x, y, edges)
                if len(bx):
                    ax.plot(bx, by, '-o', color=color, ms=4, lw=1.6)
                line += f'{spearman(x, y):>18.3f}'
            print(line)
            ax.set_title(f'{model}  (n={len(sub)})', fontsize=10)
            ax.set_xlim(-2, 102)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel(f'% of prompt notes ({args.weight})')
        for ax in axes[:, 0]:
            ax.set_ylabel(f'{METRIC_TITLES.get(metric, metric)} of generated bars')
        axes.flat[0].legend(loc='lower left', fontsize=8, framealpha=0.9)
        grp = '' if args.group == 'all' else f', {args.group} keys'
        fig.suptitle(f'prompt mix vs {METRIC_TITLES.get(metric, metric)}'
                     f'  (seeds {",".join(map(str, seeds))}{grp}; points = one prompt, '
                     f'lines = bin means)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        suffix = f'_{args.weight}' + ('' if args.group == 'all' else f'_{args.group}') \
                 + ('_pooled' if args.pool else '') + args.tag
        out = os.path.join(args.out_dir, f'prompt_mix_{metric}{suffix}.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
