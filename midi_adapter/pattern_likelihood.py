"""Does the model's distribution know which realisation the song is in?

Pattern-following came back at chance for every model, baselines included.
That has two very different explanations, and generation cannot tell them
apart:

  the model has no representation of the latent -- P(next bar) does not
  depend on what the earlier bars were, and no decoding change can help;

  it has one, and greedy/sampling washes it out -- then decoding is the fix.

So ask the distribution directly, teacher-forced, with no generation at all.
Build a context of N bars all in realisation P, splice on a final bar in
realisation Q, and score that final bar. A pattern-aware model should put its
lowest NLL on Q = P, giving a diagonal matrix. A model with no representation
gives five identical rows.

Context and candidate come from two songs that differ ONLY in pattern: same
key, same chord quality (pinned, since `walking` reads the quality's chord
tones), same bar positions. So the harmony is identical in every cell of a
row and the only thing varying is the realisation.

Scoring counts EVERY token slot of the final bar, pad and eos included, and
does not ignore_index them. That is deliberate: "no onset on beat 2" is a
prediction the model makes, and it is exactly the signal that separates
`whole` from `root4`. Ignoring pad would score one note for `whole` against
four for `root4` and make the totals incomparable.

Usage
-----
  python -m midi_adapter.pattern_likelihood \\
      --base_ckpt ckpt/.../pre.ckpt --ckpt_dir checkpoints_bar_fix \\
      --runs cp_yy_frozen_s1 --n_skip 1 --beats_per_chord 4 \\
      --rule_key_select first --max_polyphony 2 --n_context_bars 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_results_table import build_entries
from midi_adapter.evaluate_cp_bass import ALL_SEEN, UNSEEN, ROOT_NAMES
from midi_adapter.generate_synthetic_bass import BAR_PATTERNS, COMMON_CHORDS
from midi_adapter.infer_cp_bass import _prompt_from_key


@torch.no_grad()
def bar_nll(model, is_adapter, x, bar_index, bpc):
    """Total NLL of every token in bar `bar_index`, teacher-forced.

    Both CPYinyangTransformer.forward and RoFormerSymbolicTransformer.forward
    return local_decode(h, emb) of shape (B*T, subseq, vocab), where position t
    predicts x[:, t, :]. The host's own shift (sos prepended, last beat
    dropped) makes that causal, so the bar is scored on the bars before it.
    """
    B, T, S = x.shape
    logits = model(x)
    assert logits.shape[:2] == (B * T, S), \
        f'unexpected logits shape {tuple(logits.shape)}'
    logits = logits.view(B, T, S, -1)

    lo, hi = bar_index * bpc, (bar_index + 1) * bpc
    lg = logits[:, lo:hi, :, :].reshape(-1, logits.shape[-1])
    tg = x[:, lo:hi, :].reshape(-1)
    # reduction='sum' over the bar: this is log P(bar | context), and every
    # candidate bar has the same number of slots, so the totals compare.
    return float(F.cross_entropy(lg.float(), tg, reduction='sum'))


def build(key, pattern, quality, n_bars, bpc, device, base, max_poly):
    return _prompt_from_key(key, n_prompt_bars=0, device=device, base=base,
                            n_prompt_beats=n_bars * bpc,
                            max_polyphony=max_poly, cycle_unit='bar',
                            pattern=pattern, quality=quality)


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
    p.add_argument('--n_context_bars', type=int, default=4,
                   help='Bars of pattern P before the scored bar.')
    p.add_argument('--qualities', nargs='*', default=None,
                   help='Chord qualities to average over (default: all).')
    p.add_argument('--csv_out', default=None)
    p.add_argument('--device', default=None)
    # Unused by this tool but read by build_entries.
    p.add_argument('--n_prompt_beats', type=int, default=4)
    p.add_argument('--n_gen_beats',    type=int, default=28)
    p.add_argument('--temperature',    type=float, default=0.0)
    args = p.parse_args()

    if args.beats_per_chord == 1:
        raise SystemExit('Realisations exist only in --cycle_unit bar data.')

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    bpc = args.beats_per_chord
    n_bars = args.n_context_bars + 1
    scored = args.n_context_bars              # the last bar
    keys = ALL_SEEN + UNSEEN
    quals = args.qualities or list(COMMON_CHORDS)

    print('=' * 60)
    print('Loading models')
    print('=' * 60)
    print(f'  {args.n_context_bars} context bars + 1 scored bar, '
          f'{len(keys)} keys x {len(quals)} qualities')
    entries = build_entries(args, device)
    base = entries[0][1]

    print()
    print('=' * 60)
    print('Pattern likelihood  (NLL of the final bar, lower = more likely)')
    print('=' * 60)

    rows = []
    for lbl, model, is_ad in entries:
        model.eval()
        grid = {P: {Q: 0.0 for Q in BAR_PATTERNS} for P in BAR_PATTERNS}
        right = total = 0
        for P in BAR_PATTERNS:
            for key in keys:
                for q in quals:
                    ctx = build(key, P, q, n_bars, bpc, device, base,
                                args.max_polyphony)
                    per_q = {}
                    for Q in BAR_PATTERNS:
                        cand = build(key, Q, q, n_bars, bpc, device, base,
                                     args.max_polyphony)
                        x = torch.cat([ctx[:, :scored * bpc, :],
                                       cand[:, scored * bpc:, :]], dim=1)
                        per_q[Q] = bar_nll(model, is_ad, x, scored, bpc)
                        grid[P][Q] += per_q[Q]
                    right += int(min(per_q, key=per_q.get) == P)
                    total += 1

        n = len(keys) * len(quals)
        print(f'  {lbl}    argmin on the diagonal: {right}/{total} '
              f'= {right / max(total, 1):.3f}   (chance 0.200)')
        hdr = '    ' + 'context vs cand'.ljust(16) + \
              ''.join(f'{Q:>10}' for Q in BAR_PATTERNS)
        print(hdr)
        for P in BAR_PATTERNS:
            vals = {Q: grid[P][Q] / n for Q in BAR_PATTERNS}
            best = min(vals, key=vals.get)
            line = f'    {P:<16}'
            for Q in BAR_PATTERNS:
                mark = '*' if Q == best else ' '
                line += f'{vals[Q]:>9.2f}{mark}'
            print(line)
            rows.append({'model': lbl, 'context': P,
                         **{f'nll_{Q}': vals[Q] for Q in BAR_PATTERNS},
                         'argmin': best})
        # Diagonal minus the best competitor, averaged: how much the model
        # actually prefers continuing the pattern it was given.
        margins = []
        for P in BAR_PATTERNS:
            v = {Q: grid[P][Q] / n for Q in BAR_PATTERNS}
            others = [v[Q] for Q in BAR_PATTERNS if Q != P]
            margins.append(min(others) - v[P])
        print(f'    mean margin (best competitor - diagonal): '
              f'{sum(margins) / len(margins):+.2f} nats   '
              f'(positive = prefers continuing P)')
        print()

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
        with open(args.csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f'wrote {args.csv_out}')

    print('A diagonal matrix means the distribution is pattern-aware and the')
    print('failure is in decoding. Five near-identical rows mean the model has')
    print('no representation of which realisation the song is in, and no')
    print('decoding change will produce one.')


if __name__ == '__main__':
    main()
