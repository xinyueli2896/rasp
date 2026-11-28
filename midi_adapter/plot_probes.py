"""
Layer-wise figure for the X11 round 1 probes, from probe_train's summary.csv.

One panel per probe; one colour per model; solid = probe test files of the 20
trained tonic chords, dashed = the four held-out tonic chords. A band shows
+-1 s.d. over split seeds when summary.csv carries it.

    python -m midi_adapter.plot_probes --summary $WORK/probe_r1/summary.csv \\
        --out $WORK/probe_r1/probe_layers.png
"""

from __future__ import annotations

import argparse
import csv
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = [('base', 'Pretrained base (v0.42)', '#257de0'),
          ('finetune', 'Finetune baseline', '#f47230'),
          ('random', 'Random init', '#a3a3a3')]
TASKS = [('roman', 'Next roman numeral (I / IV / V)', 0.5, 'majority 0.50'),
         ('next_chord', 'Next chord (24 triads)', 1 / 24, 'chance 0.04')]
INK, INK2, GRID = '#1f1f1f', '#6b6b6b', '#e6e6e6'


def load(path):
    d = collections.defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['layer'] == '':
                continue
            def g(k):
                v = r.get(k, '')
                return float(v) if v not in ('', None) else None
            d[(r['model'], r['task'])][int(r['layer'])] = (
                g('test_acc_mean'), g('heldout_acc_mean'),
                g('test_acc_sd'), g('heldout_acc_sd'))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    d = load(args.summary)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (task, title, floor, floor_lab) in zip(axes, TASKS):
        ax.axhline(floor, color=INK2, lw=1, ls=':', zorder=1)
        ax.text(12.2, floor - 0.012, floor_lab, color=INK2, fontsize=8, va='top', ha='right')
        for model, label, col in MODELS:
            pts = d.get((model, task))
            if not pts:
                continue
            L = sorted(pts)
            for i, ls in ((0, '-'), (1, '--')):
                y = [pts[l][i] for l in L]
                sd = [pts[l][i + 2] for l in L]
                ax.plot(L, y, color=col, lw=2, ls=ls, zorder=3,
                        marker='o', ms=3.5, markeredgecolor='white', markeredgewidth=0.8)
                if all(s is not None for s in sd):
                    ax.fill_between(L, [a - s for a, s in zip(y, sd)],
                                    [a + s for a, s in zip(y, sd)],
                                    color=col, alpha=0.12, lw=0, zorder=2)
        ax.set_title(title, color=INK, fontsize=11, loc='left', pad=8)
        ax.set_xlabel('Layer (0 = previous step only, no context)')
        ax.set_xticks(range(0, 13, 2))
        ax.set_xlim(-0.3, 12.3)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('Probe accuracy')

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2) for _, _, c in MODELS]
    labels = [l for _, l, _ in MODELS]
    styles = [Line2D([], [], color=INK, lw=1.5, ls='-'),
              Line2D([], [], color=INK, lw=1.5, ls='--')]
    style_labels = ['Trained tonics (test files)', 'Held-out tonics (F#, D#m, G#, Fm)']
    fig.legend(handles, labels, loc='lower left', ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.06, 0.03))
    fig.legend(styles, style_labels, loc='lower left', ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.06, -0.03), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
