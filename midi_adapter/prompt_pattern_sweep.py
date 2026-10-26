"""Does the prompt's realisation change what the model does?

The evaluation prompt is a synthetic bar, and generate_song draws one of the
five realisations at random per call -- so twenty trials silently average over
all five and nothing reports which one was given. This pins the prompt's
pattern and sweeps it, which answers two separate questions:

  rule-following   does naming the right root get harder when the prompt is
                   `walking` (three non-root beats) than when it is `whole`
                   (one note and three rests)?

  continuation     does the model keep playing the pattern it was handed? One
                   bar identifies the pattern uniquely -- all five differ
                   inside a single bar -- so the prompt is unambiguous, and a
                   model that switches is dropping information it was given,
                   not guessing in its absence.

A generated bar is classified by its shape relative to its OWN downbeat, not
to the expected root. That deliberately separates the two questions: a bar can
be a well-formed `walking` on the wrong chord, and that counts as continuing
the pattern while failing rule-following.

Usage
-----
  python -m midi_adapter.prompt_pattern_sweep \\
      --base_ckpt ckpt/.../pre.ckpt --ckpt_dir checkpoints_bar_fix \\
      --runs cp_yy_frozen_s1 --n_skip 1 --beats_per_chord 4 \\
      --rule_key_select first --max_polyphony 2 \\
      --n_prompt_beats 4 --n_gen_beats 28 --n_trials 20 --temperature 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_results_table import build_entries
from midi_adapter.evaluate_cp_bass import (ALL_SEEN, UNSEEN, _expected_pc,
                                           _scored_positions,
                                           _gen_pcs as base_gen_pcs)
from midi_adapter.evaluate_cp_yinyang import _gen_pcs as adapter_gen_pcs

# Semitones above the bar's root; None is a beat with no onset. These are the
# five realisations generate_synthetic_bass draws from.
PATTERNS = {
    'root4':   [0, 0, 0, 0],
    'root2':   [0, None, 0, None],
    'whole':   [0, None, None, None],
    'half':    [0, None, 7, None],
    'walking': [0, 4, 7, 0],
}


def classify_bar(pcs):
    """Which of the five realisations this bar's shape is, or None.

    Measured against the bar's own downbeat, so a correctly-shaped bar on the
    wrong chord still classifies. A bar whose downbeat has no note cannot be
    referenced and counts as unclassified.
    """
    if not pcs or pcs[0] is None:
        return None
    r = pcs[0]
    steps = [None if p is None else (p - r) % 12 for p in pcs]
    for name, tmpl in PATTERNS.items():
        if steps == tmpl:
            return name
    return None


@torch.no_grad()
def trial(model, is_adapter, key, args, device, pattern):
    """(hits, scored, [bar classification]) for one generation."""
    gen = adapter_gen_pcs if is_adapter else base_gen_pcs
    bpc = args.beats_per_chord
    n_p = args.n_prompt_beats
    total = n_p + args.n_gen_beats

    pcs = gen(model, key, args.n_gen_beats, device, args.temperature, n_p,
              max_polyphony=args.max_polyphony, beats_per_chord=bpc,
              pattern=pattern)

    hits = scored = 0
    for t in _scored_positions(n_p, total, bpc):
        pc = pcs[t - n_p]
        scored += 1
        hits += int(pc is not None and pc == _expected_pc(key, t, bpc))

    # pcs[0] is absolute beat n_p, which is a bar boundary because n_p is a
    # multiple of bpc (checked in main), so bars line up with no offset.
    bars = [classify_bar(pcs[i:i + bpc])
            for i in range(0, len(pcs) - bpc + 1, bpc)]
    return hits, scored, bars


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
    p.add_argument('--n_gen_beats',    type=int, default=28)
    p.add_argument('--n_trials',       type=int, default=20)
    p.add_argument('--temperature',    type=float, default=0.0)
    p.add_argument('--csv_out', default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.beats_per_chord == 1:
        raise SystemExit('Realisation patterns exist only in --cycle_unit bar '
                         'data; --beats_per_chord 1 has one note per chord.')
    if args.n_prompt_beats % args.beats_per_chord:
        raise SystemExit(
            f'--n_prompt_beats {args.n_prompt_beats} is not a whole number of '
            f'bars at --beats_per_chord {args.beats_per_chord}; generated bars '
            f'would not line up with the prompt.')

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    keys_all = ALL_SEEN + UNSEEN

    print('=' * 60)
    print('Loading models')
    print('=' * 60)
    print(f'  protocol : {args.n_prompt_beats}-beat prompt '
          f'({args.n_prompt_beats // args.beats_per_chord} bar), '
          f'{args.n_gen_beats} generated, temperature {args.temperature}, '
          f'{args.n_trials} trials')
    entries = build_entries(args, device)

    # None sweeps last, as the control: it is the random-pattern prompt every
    # earlier table used.
    sweep = list(PATTERNS) + [None]

    print()
    print('=' * 60)
    print('Sweeping prompt patterns')
    print('=' * 60)

    rows = []
    for lbl, model, is_ad in entries:
        print(f'  {lbl}')
        for pat in sweep:
            h = s = 0
            hu = su = 0
            cont = wf = nbars = 0
            cons_h = cons_n = 0
            for key in keys_all:
                for _ in range(args.n_trials):
                    hits, scored, bars = trial(model, is_ad, key, args,
                                               device, pat)
                    h += hits; s += scored
                    if key in UNSEEN:
                        hu += hits; su += scored
                    for b in bars:
                        nbars += 1
                        wf += int(b is not None)
                        cont += int(pat is not None and b == pat)
                    # Self-consistency: does the model hold ONE realisation for
                    # the whole generation, whichever it picked? The training
                    # data draws one pattern per song, so this is a property of
                    # the data the model either reproduces or does not -- and
                    # unlike `continues` it needs no pinned prompt.
                    if bars and bars[0] is not None:
                        cons_h += sum(1 for b in bars if b == bars[0])
                        cons_n += len(bars)
            rows.append({
                'model': lbl,
                'prompt': pat or 'random',
                'acc_all': h / max(s, 1),
                'acc_unseen': hu / max(su, 1),
                'continues': (cont / nbars) if (pat and nbars) else float('nan'),
                'consistent': cons_h / max(cons_n, 1),
                'wellformed': wf / max(nbars, 1),
            })
            r = rows[-1]
            cstr = ('  --  ' if pat is None else f'{r["continues"]:.3f}')
            print(f'    {r["prompt"]:<9} acc {r["acc_all"]:.3f}  '
                  f'unseen {r["acc_unseen"]:.3f}  |  continues {cstr}  '
                  f'consistent {r["consistent"]:.3f}  '
                  f'well-formed {r["wellformed"]:.3f}')

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
        with open(args.csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f'\nwrote {args.csv_out}')

    print()
    print('Rule-following   acc / unseen  -- is the downbeat the right root?')
    print('Pattern-following')
    print("  continues      bars matching the PROMPT's pattern (needs a pinned prompt)")
    print('  consistent     bars matching this generation\'s OWN first bar, i.e. does')
    print('                 it hold one realisation the way the training data does')
    print('  well-formed    bars matching ANY of the five realisations')
    print("All three are measured against each bar's own downbeat, so they are "
          "independent\nof whether the root is correct: a clean `walking` on the "
          "wrong chord counts as\npattern-following and fails rule-following.")


if __name__ == '__main__':
    main()
