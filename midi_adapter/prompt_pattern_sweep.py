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
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_results_table import build_entries
from midi_adapter.evaluate_cp_bass import (ALL_SEEN, UNSEEN, _expected_pc,
                                           _scored_positions, _extract_pc,
                                           _gen_pcs as base_gen_pcs)
from midi_adapter.evaluate_cp_bass import _save_midi_all_keys as save_base_midi
from midi_adapter.evaluate_cp_yinyang import _gen_pcs as adapter_gen_pcs
from midi_adapter.evaluate_cp_yinyang import _save_midi_all_keys as save_adapter_midi
from midi_adapter.bar_patterns import PATTERNS, classify_bar
from midi_adapter.generate_synthetic_bass import MIXED_PATTERN
from midi_adapter.infer_cp_bass import _prompt_from_key

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

    # pcs[0] is absolute beat n_p, a bar boundary because n_p is a multiple of
    # bpc (checked in main), so bars line up with no offset.
    bars = [classify_bar(pcs[i:i + bpc])
            for i in range(0, len(pcs) - bpc + 1, bpc)]
    return hits, scored, bars


@torch.no_grad()
def data_trial(base, key, args, device, pattern):
    """The same measurement on the GROUND-TRUTH data instead of a generation.

    This is the ceiling, and it is not optional. Every number here must come
    out at 1.000: the data follows the rule by construction and holds one
    realisation per song. Anything less means the classifier or the pitch-class
    extraction is wrong, and the model rows below it mean nothing.
    """
    bpc = args.beats_per_chord
    n_p = args.n_prompt_beats
    total = n_p + args.n_gen_beats

    x = _prompt_from_key(key, n_prompt_bars=0, device=device, base=base,
                         n_prompt_beats=total,
                         max_polyphony=args.max_polyphony,
                         cycle_unit='bar', pattern=pattern)
    pcs = [_extract_pc(x[:, t, :], base.tokenizer) for t in range(n_p, total)]

    hits = scored = 0
    for t in _scored_positions(n_p, total, bpc):
        pc = pcs[t - n_p]
        scored += 1
        hits += int(pc is not None and pc == _expected_pc(key, t, bpc))
    bars = [classify_bar(pcs[i:i + bpc])
            for i in range(0, len(pcs) - bpc + 1, bpc)]
    return hits, scored, bars


def print_lift(rows, model_label):
    """How much the prompt moved the output, net of what the model does anyway.

    `continues` alone is confounded: root4 scores highest partly because root4
    is what these models emit whatever they are shown, which is the same
    reason `whole` reads 0.000. Lift subtracts that prior:

        lift(X) = P(bar = X | prompt = X) - mean_(Y != X) P(bar = X | prompt = Y)

    Zero means the prompt changed nothing. Positive means the prompt's
    realisation actually carried.
    """
    mine = [r for r in rows if r['model'] == model_label and r['prompt'] in PATTERNS]
    if len(mine) < 2:
        return
    out = []
    for r in mine:
        x = r['prompt']
        others = [o[f'p_{x}'] for o in mine if o['prompt'] != x]
        r['lift'] = r[f'p_{x}'] - sum(others) / len(others)
        out.append(f'{x}={r["lift"]:+.3f}')
    print('              lift:     ' + '  '.join(out))


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
    p.add_argument('--save_midi_dir', default=None,
                   help='Write one MIDI per key per prompt pattern, so the '
                        'numbers can be listened to: '
                        '<dir>/<model>/<prompt>/key_<name>.mid')
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
    sweep = list(PATTERNS) + [MIXED_PATTERN, None]

    print()
    print('=' * 60)
    print('Sweeping prompt patterns')
    print('=' * 60)

    # (label, model, is_adapter) plus a sentinel for the ground-truth ceiling.
    DATA = '__data__'
    rows = []
    for lbl, model, is_ad in [('Data (ceiling)', entries[0][1], DATA)] + entries:
        print(f'  {lbl}')
        for pat in sweep:
            h = s = 0
            hu = su = 0
            cont = wf = nbars = 0
            cons_h = cons_n = 0
            hist: Counter = Counter()
            for key in keys_all:
                for _ in range(args.n_trials):
                    if is_ad == DATA:
                        if pat is None:
                            continue        # "random" is meaningless for data
                        hits, scored, bars = data_trial(model, key, args,
                                                        device, pat)
                    else:
                        hits, scored, bars = trial(model, is_ad, key, args,
                                                   device, pat)
                    h += hits; s += scored
                    if key in UNSEEN:
                        hu += hits; su += scored
                    for b in bars:
                        nbars += 1
                        wf += int(b is not None)
                        cont += int(pat in PATTERNS and b == pat)
                        hist[b or 'none'] += 1
                    # Self-consistency: does the model hold ONE realisation for
                    # the whole generation, whichever it picked? The training
                    # data draws one pattern per song, so this is a property of
                    # the data the model either reproduces or does not -- and
                    # unlike `continues` it needs no pinned prompt.
                    if bars and bars[0] is not None:
                        cons_h += sum(1 for b in bars if b == bars[0])
                        cons_n += len(bars)
            if is_ad == DATA and pat is None:
                continue
            wfr = wf / max(nbars, 1)
            # What `consistent` reads if each bar's realisation is drawn
            # independently: the first bar always matches itself, and each
            # later one matches with probability (well-formed / 5). Printed
            # because 0.3 is meaningless until you know chance is also 0.3.
            nb = max(1, args.n_gen_beats // args.beats_per_chord)
            chance = (1 + (nb - 1) * wfr / len(PATTERNS)) / nb
            rows.append({
                'model': lbl,
                'prompt': pat or 'random',
                'acc_all': h / max(s, 1),
                'acc_unseen': hu / max(su, 1),
                'continues': (cont / nbars) if (pat in PATTERNS and nbars)
                             else float('nan'),
                'consistent': cons_h / max(cons_n, 1),
                'consistent_chance': chance,
                'wellformed': wfr,
                **{f'p_{k}': hist[k] / max(nbars, 1)
                   for k in list(PATTERNS) + ['none']},
            })
            r = rows[-1]
            cstr = (f'{r["continues"]:.3f}' if pat in PATTERNS else '  --  ')
            print(f'    {r["prompt"]:<9} acc {r["acc_all"]:.3f}  '
                  f'unseen {r["acc_unseen"]:.3f}  |  continues {cstr}  '
                  f'consistent {r["consistent"]:.3f} (chance {chance:.3f})  '
                  f'well-formed {r["wellformed"]:.3f}')
            print('              produced: ' + '  '.join(
                f'{k}={hist[k] / max(nbars, 1):.2f}'
                for k in list(PATTERNS) + ['none'] if hist[k]))

            if args.save_midi_dir and pat is not None:
                saver = save_adapter_midi if is_ad is True else save_base_midi
                if is_ad != DATA:
                    out = os.path.join(args.save_midi_dir, lbl, pat)
                    saver(model, args.n_gen_beats, device, args.temperature,
                          args.n_prompt_beats, out,
                          max_polyphony=args.max_polyphony,
                          beats_per_chord=args.beats_per_chord,
                          pattern=pat, quiet=True)

        print_lift(rows, lbl)

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
        with open(args.csv_out, 'w', newline='') as f:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(f, fieldnames=fields, restval='')
            w.writeheader(); w.writerows(rows)
        print(f'\nwrote {args.csv_out}')

    # Only the PINNED Data rows should read 1.000 throughout. The `mixed` row
    # redraws per bar, so its `consistent` is correctly at chance and its
    # `continues` is undefined -- checking those made the gate cry wolf.
    bad = [r for r in rows if r['model'].startswith('Data')
           and ((r['prompt'] in PATTERNS
                 and min(r['acc_all'], r['continues'], r['consistent'],
                         r['wellformed']) < 0.999)
                or (r['prompt'] == MIXED_PATTERN
                    and min(r['acc_all'], r['wellformed']) < 0.999))]
    if bad:
        print()
        print('!! The Data (ceiling) rows are not 1.000. The classifier or the')
        print('!! pitch-class extraction is wrong and every model row below is')
        print('!! uninterpretable. Fix that before reading anything else.')

    print()
    print('Rule-following   acc / unseen  -- is the downbeat the right root?')
    print('Pattern-following')
    print("  continues      bars matching the PROMPT's pattern (needs a pinned prompt)")
    print('  consistent     bars matching this generation\'s OWN first bar, i.e. does')
    print('                 it hold one realisation the way the training data does')
    print('  well-formed    bars matching ANY of the five realisations')
    print('  produced       what the generated bars actually were')
    print('  lift           P(X | prompt=X) - mean P(X | prompt!=X). continues')
    print("                 is confounded by the model's own prior; lift is not.")
    print("All three are measured against each bar's own downbeat, so they are "
          "independent\nof whether the root is correct: a clean `walking` on the "
          "wrong chord counts as\npattern-following and fails rule-following.")


if __name__ == '__main__':
    main()
