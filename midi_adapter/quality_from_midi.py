"""
What did a v5 model do about chord QUALITY, when nothing told it to?

v5's rule model emits a root per bar and nothing else, so a v5 run carries no
instruction about major or minor. Its generated thirds are therefore a free
observation of where quality comes from when the rule is silent, and that
predicts how much work v6's quality channel has to do. Three outcomes, three
different conclusions:

  follows the prompt   the host already reads mode from context, and v6's
                       channel mostly confirms what it would do anyway
  defaults to major    the parallel-major trap is real -- a model reading the
                       tonic right and the mode wrong scores 1.000 on every
                       v4/v5 metric, so v6 is necessary to see it at all
  samples its prior    it ignores the prompt and reproduces the corpus mix,
                       which means the mode has to be supplied, not inferred

It also answers the leakage question for an unseen tonic directly: on a key the
adapter never trained on, does it hold the prompt's mode or fall back?

Reads the MIDI an eval already wrote, so it needs no checkpoints and no GPU:

    python -I midi_adapter/quality_from_midi.py --midi_dir <eval_midi_v5>

Quality of a bar is the duration-weighted mass on the major third against the
minor third, measured relative to that bar's RULE root -- the same definition
census_v6_quality uses on the data, so the two are comparable. Weights are
clipped at the bar line, as the evaluator's scorer does.

One difference from the census, unavoidable here: MIDI keeps no slot order, so
"the bass" is the lowest note starting on a step rather than slot 0. The
lowest onset of each step is excluded by default, since a bass passing through
the third would otherwise decide a bar.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Deliberately NOT importing prompt_mix_from_midi or evaluate_cp_bass: both
# reach cp_transformer and so the whole training stack, for two constants and
# twelve lines of MIDI reading. This is a read-only analysis of files an eval
# already wrote, and it should run anywhere pretty_midi does -- a laptop with
# no torch, for instance. OFFSETS still comes from the one canonical place.
from rasp_program.sequence_rule import OFFSETS

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEY_OF = {n: i for i, n in enumerate(ROOT_NAMES)}
SECONDS_PER_SUBBEAT = 0.125       # v4_eval.write_midi: 120 BPM, beat_div 4

MAJ, MIN, AMB, NONE = 'M', 'm', '?', '.'
DURATION_CLIP = True


def notes_on_grid(path: str):
    """[(step, dur_steps, pitch)] of every non-drum note, on the subbeat grid.

    The same reading prompt_mix_from_midi does; kept here so this script does
    not pull that module's dependencies in behind it.
    """
    out = []
    for ins in pretty_midi.PrettyMIDI(path).instruments:
        if ins.is_drum:
            continue
        for n in ins.notes:
            step = int(round(n.start / SECONDS_PER_SUBBEAT))
            dur = max(1, int(round((n.end - n.start) / SECONDS_PER_SUBBEAT)))
            out.append((step, dur, n.pitch))
    return out


def bar_quality(notes, lo: int, hi: int, root: int, *, exclude_bass: bool,
                margin: float):
    """Quality of the bar covering subbeats [lo, hi): 'M', 'm', '?' or '.'."""
    by_step: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for step, dur, pitch in notes:
        if lo <= step < hi:
            by_step[step].append((pitch, dur))
    maj = mina = 0.0
    for step, items in by_step.items():
        if exclude_bass and len(items) > 1:
            items = sorted(items)[1:]          # drop the lowest onset
        elif exclude_bass:
            continue                           # a lone note IS the bass
        for pitch, dur in items:
            w = min(dur, hi - step) if DURATION_CLIP else dur
            rel = (pitch - root) % 12
            if rel == 4:
                maj += w
            elif rel == 3:
                mina += w
    if maj <= 0 and mina <= 0:
        return NONE
    if maj > 0 and mina > 0 and min(maj, mina) > margin * max(maj, mina):
        return AMB
    return MAJ if maj > mina else MIN


def analyse(path: str, key: int, *, n_prompt: int, spc: int, n_bars: int,
            exclude_bass: bool, margin: float):
    """The prompt bar's quality and each generated bar's, for one file."""
    notes = notes_on_grid(path)
    quals = []
    for b in range(n_bars):
        lo = b * spc
        root = (key + OFFSETS[b % len(OFFSETS)]) % 12
        quals.append(bar_quality(notes, lo, lo + spc, root,
                                 exclude_bass=exclude_bass, margin=margin))
    n_prompt_bars = max(1, n_prompt // spc)
    return quals[:n_prompt_bars], quals[n_prompt_bars:]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--midi_dir', required=True,
                   help='an eval MIDI tree: <dir>/s<seed>/p<P>_T<t>/<model>/key_*.mid')
    p.add_argument('--protocol', default='p16_T0')
    p.add_argument('--spc', type=int, default=16, help='subbeats per bar')
    p.add_argument('--n_prompt', type=int, default=16)
    p.add_argument('--n_bars', type=int, default=8)
    p.add_argument('--margin', type=float, default=0.25)
    p.add_argument('--include_bass', action='store_true',
                   help='let the lowest onset of each step vote on the harmony')
    p.add_argument('--unseen', type=int, nargs='*', default=[6, 8],
                   help='tonics held out of finetuning; v4/v5 used F# and G#')
    p.add_argument('--models', nargs='*', default=None)
    p.add_argument('--csv_out', default=None)
    args = p.parse_args()

    files = []
    for dirpath, dirnames, filenames in os.walk(args.midi_dir):
        dirnames[:] = [d for d in dirnames if d != '__MACOSX']
        if os.path.basename(os.path.dirname(dirpath)) != args.protocol:
            continue
        model = re.sub(r'_s\d+$', '', os.path.basename(dirpath))
        seed = re.search(r'(?:^|/)s(\d+)/', dirpath + '/')
        for fn in sorted(filenames):
            m = re.fullmatch(r'key_([A-G]#?)(?:_t(\d+))?\.mid', fn)
            if m and not fn.startswith('.'):
                files.append((model, int(seed.group(1)) if seed else -1,
                              KEY_OF[m.group(1)], int(m.group(2) or 0),
                              os.path.join(dirpath, fn)))
    if not files:
        sys.exit(f'no key_*.mid under {args.midi_dir}/*/{args.protocol}/*/')
    models = args.models or sorted({m for m, *_ in files},
                                   key=lambda s: (s not in ('Data',), s))
    print(f'{len(files)} files, {len(models)} models, protocol {args.protocol}')
    print(f'quality from {"every note" if args.include_bass else "the non-bass onsets"}'
          f', relative to each bar\'s RULE root\n')

    rows = []
    for model, seed, key, trial, path in files:
        pq, gq = analyse(path, key, n_prompt=args.n_prompt, spc=args.spc,
                         n_bars=args.n_bars, exclude_bass=not args.include_bass,
                         margin=args.margin)
        rows.append({'model': model, 'seed': seed, 'key': key,
                     'key_name': ROOT_NAMES[key], 'trial': trial,
                     'unseen': key in args.unseen,
                     'prompt': pq[0] if pq else NONE,
                     'gen': ''.join(gq)})

    # --- does the generation hold the prompt's mode? ----------------------
    print('P(generated third | prompt third), over generated bars whose third '
          'is voiced at all')
    print(f'{"model":<26}{"prompt M -> M":>15}{"prompt m -> m":>15}'
          f'{"gen major rate":>16}{"gen unvoiced":>14}')
    for model in models:
        sub = [r for r in rows if r['model'] == model]
        tally = {MAJ: collections.Counter(), MIN: collections.Counter()}
        unvoiced = total = 0
        for r in sub:
            for q in r['gen']:
                total += 1
                if q in (NONE, AMB):
                    unvoiced += 1
                if r['prompt'] in tally and q in (MAJ, MIN):
                    tally[r['prompt']][q] += 1
        def rate(pm, want):
            n = sum(tally[pm].values())
            return tally[pm][want] / n if n else float('nan')
        gen_maj = sum(t[MAJ] for t in tally.values())
        gen_all = sum(sum(t.values()) for t in tally.values())
        print(f'{model:<26}{rate(MAJ, MAJ):>15.3f}{rate(MIN, MIN):>15.3f}'
              f'{(gen_maj / gen_all if gen_all else float("nan")):>16.3f}'
              f'{unvoiced / max(total, 1):>14.3f}')

    # --- seen vs unseen keys ---------------------------------------------
    print('\nthe same, split by whether the tonic was trained on '
          f'(unseen: {[ROOT_NAMES[k] for k in args.unseen]})')
    print(f'{"model":<26}{"seen m -> m":>14}{"unseen m -> m":>15}'
          f'{"seen M -> M":>14}{"unseen M -> M":>15}')
    for model in models:
        out = [f'{model:<26}']
        for want, pm in ((MIN, MIN), (MAJ, MAJ)):
            for unseen in (False, True):
                sub = [r for r in rows if r['model'] == model
                       and r['unseen'] == unseen and r['prompt'] == pm]
                hits = sum(q == want for r in sub for q in r['gen'])
                n = sum(q in (MAJ, MIN) for r in sub for q in r['gen'])
                out.append(f'{(hits / n if n else float("nan")):>14.3f}'
                           if not unseen else
                           f'{(hits / n if n else float("nan")):>15.3f}')
        print(''.join(out))

    # --- bar 2 of the cycle, where harmonic minor flips ------------------
    print('\nbar 2 of each cycle, on MINOR prompts: the rule would make it '
          'MAJOR (the V)')
    print(f'{"model":<26}{"bar2 major":>12}{"bar2 minor":>12}'
          f'{"bar2 unvoiced":>15}{"n":>7}')
    for model in models:
        c = collections.Counter()
        for r in rows:
            if r['model'] != model or r['prompt'] != MIN:
                continue
            # generated bars start at bar 1, so cycle bar 2 sits at gen index 1
            for i, q in enumerate(r['gen']):
                if (i + 1) % len(OFFSETS) == 2:
                    c[q] += 1
        n = sum(c.values())
        print(f'{model:<26}{c[MAJ] / max(n,1):>12.3f}{c[MIN] / max(n,1):>12.3f}'
              f'{(c[NONE] + c[AMB]) / max(n,1):>15.3f}{n:>7}')

    print('\nprompt quality as the eval presented it (the Data row is the '
          'corpus itself)')
    for model in models[:3]:
        c = collections.Counter(r['prompt'] for r in rows if r['model'] == model)
        tot = sum(c.values())
        print(f'  {model:<24}' + '  '.join(
            f'{k}={c[k] / max(tot,1):.3f}' for k in (MAJ, MIN, AMB, NONE)))

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
        with open(args.csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.csv_out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
