"""Rule-following accuracy per CYCLE, as the generation runs on.

The table reports one number per key group over the whole generated span,
which cannot distinguish two very different failures:

  immediate   the first generated bar is already wrong and stays wrong
  drift       early bars are right and error accumulates as the model
              consumes its own output

Both read as "low accuracy". Plotted against cycle index they are a flat
line and a decaying one.

Why a CYCLE (4 bars) and not a bar: a cycle is the rule's period, so it
holds exactly one bar of each phase. Per-bar would show a sawtooth from
phase-dependent difficulty -- OFFSETS is [0,5,7,0], so phases 0 and 3 both
ask for the root while 1 and 2 do not -- and that is not separable from
drift. Per-cycle is phase-balanced, so movement along x is drift alone.

Error bars are Wilson intervals over keys x trials, not +-SEM: many cells
sit at exactly 0.0 or 1.0, where the normal approximation collapses to zero
width and claims a certainty the data does not support.

The training window matters here. train_length beats is all the model ever
saw in one piece, so everything past that boundary is length extrapolation
rather than error accumulation. It is marked in the output; a cliff exactly
there is a different problem with a different fix.

Usage
-----
  python -m midi_adapter.drift_curve \\
      --base_ckpt ckpt/.../pre.ckpt --ckpt_dir checkpoints_bar_fix \\
      --runs cp_yy_frozen_s1 cp_yy_trainar_s1 \\
      --n_skip 1 --beats_per_chord 4 --rule_key_select first \\
      --max_polyphony 2 --n_prompt_beats 4 --n_gen_beats 60 \\
      --n_trials 20 --out_prefix logs_bar_fix/drift_s1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_results_table import build_entries
from midi_adapter.evaluate_cp_bass import (CATEGORIES, _expected_pc,
                                           _gen_pcs as base_gen_pcs)
from midi_adapter.evaluate_cp_yinyang import _gen_pcs as adapter_gen_pcs


def wilson(k: int, n: int, z: float = 1.96):
    """(point, lo, hi) for k successes in n Bernoulli trials."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def score_sequence(pcs, key: int, n_prompt: int, bpc: int):
    """[(cycle_index, correct)] for the scored beats of one generation.

    pcs[i] is the pitch class of absolute beat n_prompt + i, or None when the
    beat emitted no note -- which counts as wrong, never as a free pass.
    """
    out = []
    cycle_beats = bpc * 4
    for i, pc in enumerate(pcs):
        t = n_prompt + i
        if bpc > 1 and t % bpc:
            continue                      # off-beats may be chord tones
        out.append((t // cycle_beats,
                    pc is not None and pc == _expected_pc(key, t, bpc)))
    return out


@torch.no_grad()
def collect(model, is_adapter, keys, args, device):
    """{cycle: [hits, total]} pooled over keys x trials for one model."""
    gen = adapter_gen_pcs if is_adapter else base_gen_pcs
    acc: dict[int, list[int]] = {}
    for key in keys:
        for _ in range(args.n_trials):
            pcs = gen(model, key, args.n_gen_beats, device, args.temperature,
                      args.n_prompt_beats,
                      max_polyphony=args.max_polyphony,
                      beats_per_chord=args.beats_per_chord)
            for cyc, ok in score_sequence(pcs, key, args.n_prompt_beats,
                                          args.beats_per_chord):
                slot = acc.setdefault(cyc, [0, 0])
                slot[0] += int(ok)
                slot[1] += 1
    return acc


def plot(rows, args, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed -- CSV written, no figure)')
        return

    groups = [g for g, _, _ in CATEGORIES]
    models = sorted({r['model'] for r in rows}, key=lambda m: m != 'Pretrain')
    cycle_beats = args.beats_per_chord * 4
    boundary = args.train_length / cycle_beats

    for logy, suffix, ylab in ((False, '', 'rule-following accuracy'),
                               (True, '_error', 'error (1 - accuracy)')):
        fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6),
                                 sharey=True)
        for ax, g in zip(axes, groups):
            for m in models:
                pts = sorted((r['cycle'], r) for r in rows
                             if r['model'] == m and r['group'] == g)
                if not pts:
                    continue
                x = [c for c, _ in pts]
                if logy:
                    y = [max(1e-3, 1 - r['acc']) for _, r in pts]
                    lo = [max(1e-3, 1 - r['hi']) for _, r in pts]
                    hi = [max(1e-3, 1 - r['lo']) for _, r in pts]
                else:
                    y = [r['acc'] for _, r in pts]
                    lo = [r['lo'] for _, r in pts]
                    hi = [r['hi'] for _, r in pts]
                err = [[a - b for a, b in zip(y, lo)],
                       [b - a for a, b in zip(y, hi)]]
                ax.errorbar(x, y, yerr=err, marker='o', capsize=3,
                            linewidth=1.4, markersize=4, label=m)
            # Past this, the model is generating longer than anything it was
            # trained on, so a drop here is extrapolation rather than drift.
            if boundary <= max((r['cycle'] for r in rows), default=0):
                ax.axvline(boundary - 0.5, color='k', linestyle=':', linewidth=1)
            if logy:
                ax.set_yscale('log')
            ax.set_title(g, fontsize=10)
            ax.set_xlabel('cycle')
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylab)
        axes[-1].legend(fontsize=7, loc='best')
        fig.suptitle(f'{ylab} per {cycle_beats}-beat cycle '
                     f'(T={args.temperature}, {args.n_trials} trials; '
                     f'dotted = train window)', fontsize=10)
        fig.tight_layout()
        p = f'{path}{suffix}.png'
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'  wrote {p}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base_ckpt', required=True)
    p.add_argument('--ft_ckpt',   default=None)
    p.add_argument('--ckpt_dir',  default='checkpoints_cp')
    p.add_argument('--runs',      nargs='*', default=[])
    p.add_argument('--model_size',   type=int, default=1)
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip',       type=int, default=1)
    p.add_argument('--max_polyphony',   type=int, default=2)
    p.add_argument('--beats_per_chord', type=int, default=4)
    p.add_argument('--rule_key_select', default='first',
                   choices=['phase0', 'first'])
    p.add_argument('--n_prompt_beats', type=int, default=4)
    # 60 generated + 4 prompt = 64 beats = 16 bars = 4 cycles at bpc=4.
    p.add_argument('--n_gen_beats',    type=int, default=60)
    p.add_argument('--n_trials',       type=int, default=20)
    p.add_argument('--temperature',    type=float, default=0.0)
    p.add_argument('--train_length',   type=int, default=32,
                   help='Beats the model was trained on in one window; drawn '
                        'as the extrapolation boundary.')
    p.add_argument('--out_prefix', default='drift')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    cycle_beats = args.beats_per_chord * 4
    total = args.n_prompt_beats + args.n_gen_beats

    print('=' * 60)
    print('Loading models')
    print('=' * 60)
    print(f'  protocol : {args.n_prompt_beats}-beat prompt, '
          f'{args.n_gen_beats} generated, temperature {args.temperature}')
    print(f'  cycle    : {cycle_beats} beats, '
          f'{total // cycle_beats} cycles; train window = '
          f'{args.train_length / cycle_beats:.1f} cycles')
    entries = build_entries(args, device)

    print()
    print('=' * 60)
    print('Generating')
    print('=' * 60)
    rows = []
    for lbl, model, is_ad in entries:
        print(f'  {lbl} ...', end='  ', flush=True)
        for g, keys, _ in CATEGORIES:
            for cyc, (hits, n) in sorted(collect(model, is_ad, keys, args,
                                                 device).items()):
                acc, lo, hi = wilson(hits, n)
                rows.append({'model': lbl, 'group': g, 'cycle': cyc,
                             'hits': hits, 'total': n,
                             'acc': acc, 'lo': lo, 'hi': hi})
        print('done')

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    with open(f'{args.out_prefix}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f'{args.out_prefix}.json', 'w') as f:
        json.dump({'protocol': vars(args), 'rows': rows}, f, indent=2,
                  default=str)
    print()
    print(f'  wrote {args.out_prefix}.csv / .json')
    plot(rows, args, args.out_prefix)

    # Text fallback, so the shape is readable without the figure.
    print()
    cycles = sorted({r['cycle'] for r in rows})
    for g, _, _ in CATEGORIES:
        print(f'--- {g} ---')
        hdr = f'{"model":<28}' + ''.join(f' {"cyc" + str(c):>12}' for c in cycles)
        print(hdr)
        for m in sorted({r['model'] for r in rows}):
            line = f'{m:<28}'
            for c in cycles:
                hit = [r for r in rows
                       if r['model'] == m and r['group'] == g and r['cycle'] == c]
                line += (f' {hit[0]["acc"]:>5.3f}±{(hit[0]["hi"]-hit[0]["lo"])/2:<5.3f}'
                         if hit else f' {"--":>12}')
            print(line)
        print()


if __name__ == '__main__':
    main()
