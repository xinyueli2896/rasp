"""
Prompt-mix rows from the MIDI an eval already wrote, no regeneration.

v4_eval --save_midi_dir writes <dir>/s<seed>/p<P>_T<t>/<model>/key_<root>[_t<i>].mid,
each file the prompt bar(s) followed by the continuation, at 0.125 s per
subbeat. This script reads those files back onto the subbeat grid and
recomputes, per file, what --per_prompt_csv would have recorded:

  prompt mix   share of the prompt's notes on the rule root, on the third
               or fifth, on non-chord tones (by count, by duration, bass only)
  outcomes     root_exact, root_chordtone, bass_exact, bass_chordtone,
               onsets_per_bar over the generated bars, exactly as the
               evaluator's scorer (duration-weighted histograms clipped at
               the bar line, triad template root, silent bars count as wrong)
  acc          the downbeat metric. The evaluator reads slot 0 of the
               downbeat step; MIDI keeps no slot order, so here it is the
               LOWEST note starting on the downbeat. In the data slot 0 is
               the bass and the adapters learn that order, so the two agree
               on nearly every bar; the column is named acc all the same and
               the log says how it was read.

One CSV per seed, named like the evaluator's, so plot_prompt_mix.py pools them:

    python -m midi_adapter.prompt_mix_from_midi ~/Downloads/eval_midi_v5 \\
        --out_dir results_v5/prompt_mix   [--protocol p16_T0]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi_adapter.evaluate_cp_bass import UNSEEN
from midi_adapter.generate_synthetic_bass import OFFSETS, ROOT_NAMES

SECONDS_PER_SUBBEAT = 0.125      # v4_eval.write_midi: 120 BPM, beat_div 4
KEY_OF = {n: i for i, n in enumerate(ROOT_NAMES)}


def notes_on_grid(path: str):
    """[(step, dur_steps, pitch)] of every non-drum note, on the subbeat grid."""
    pm = pretty_midi.PrettyMIDI(path)
    out = []
    for ins in pm.instruments:
        if ins.is_drum:
            continue
        for n in ins.notes:
            step = int(round(n.start / SECONDS_PER_SUBBEAT))
            dur = max(1, int(round((n.end - n.start) / SECONDS_PER_SUBBEAT)))
            out.append((step, dur, n.pitch))
    return out


def infer_root(hist: np.ndarray) -> int:
    r = np.arange(12)
    score = hist + 0.5 * hist[(r + 7) % 12] + 0.5 * np.maximum(hist[(r + 3) % 12], hist[(r + 4) % 12])
    return int(score.argmax())


def tiers(hists: np.ndarray, roots: np.ndarray):
    """(n_bars, 12) histograms -> (exact, chordtone) over the bars."""
    ex = ct = 0
    for h, gt in zip(hists, roots):
        if h.sum() <= 0:
            continue
        rel = (infer_root(h) - gt) % 12
        ex += rel == 0
        ct += rel in (0, 3, 4, 7)
    return ex / len(hists), ct / len(hists)


def score_file(path: str, key: int, n_prompt: int, spc: int, n_gen: int) -> dict:
    notes = notes_on_grid(path)
    total = n_prompt + n_gen
    # lowest note starting at each step (the bass line)
    lowest = {}
    for step, dur, pitch in notes:
        if step < total and (step not in lowest or pitch < lowest[step][1]):
            lowest[step] = (dur, pitch)

    def rule_root(step):
        return (key + OFFSETS[(step // spc) % 4]) % 12

    # --- prompt mix -------------------------------------------------------
    def cls(step, pitch):
        rel = (pitch % 12 - rule_root(step)) % 12
        return 0 if rel == 0 else 1 if rel in (3, 4, 7) else 2

    cnt = np.zeros(3); dw = np.zeros(3); bass = np.zeros(3)
    n_notes = 0; steps = set()
    for step, dur, pitch in notes:
        if step >= n_prompt:
            continue
        c = cls(step, pitch)
        cnt[c] += 1; dw[c] += dur; n_notes += 1; steps.add(step)
    for step in steps:                       # one bass note per step
        bass[cls(step, lowest[step][1])] += 1
    row = {}
    for name, v in (('p_', cnt), ('pdur_', dw), ('pbass_', bass)):
        tot = max(v.sum(), 1e-9)
        row.update({f'{name}root': v[0] / tot, f'{name}chord': v[1] / tot,
                    f'{name}nct': v[2] / tot})
    row['p_notes'] = float(n_notes)
    row['p_steps_sounding'] = float(len(steps))

    # --- generated bars ---------------------------------------------------
    n_bars = n_gen // spc
    hist = np.zeros((n_bars, 12)); hist_b = np.zeros((n_bars, 12))
    onsets = 0
    for step, dur, pitch in notes:
        if step < n_prompt or step >= total:
            continue
        b = (step - n_prompt) // spc
        w = min(dur, spc - step % spc)                  # clipped at the bar line
        hist[b, pitch % 12] += w
    for step in range(n_prompt, total):
        if step in lowest:                              # one bass note per step
            onsets += 1
            dur, pitch = lowest[step]
            hist_b[(step - n_prompt) // spc, pitch % 12] += min(dur, spc - step % spc)
    roots = np.array([rule_root(n_prompt + b * spc) for b in range(n_bars)])
    row['acc'] = float(np.mean([
        (n_prompt + b * spc) in lowest
        and lowest[n_prompt + b * spc][1] % 12 == roots[b] for b in range(n_bars)]))
    row['root_exact'], row['root_chordtone'] = tiers(hist, roots)
    row['bass_exact'], row['bass_chordtone'] = tiers(hist_b, roots)
    row['onsets_per_bar'] = onsets / n_bars
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('midi_root', help='the --save_midi_dir of the eval (holds s<seed>/)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--protocol', default='p16_T0', help='subdirectory under each seed')
    ap.add_argument('--beats_per_chord', type=int, default=16)
    ap.add_argument('--n_gen_beats', type=int, default=112)
    args = ap.parse_args()
    n_prompt = int(re.match(r'p(\d+)', args.protocol).group(1))
    os.makedirs(args.out_dir, exist_ok=True)

    seed_dirs = sorted(glob.glob(os.path.join(args.midi_root, 's*', args.protocol)))
    if not seed_dirs:
        sys.exit(f'no {args.protocol} directories under {args.midi_root}/s*/')
    print(f'downbeat acc here = lowest note starting on the downbeat equals the rule root '
          f'(the evaluator reads slot 0; MIDI has no slots)')
    for sd_dir in seed_dirs:
        seed = int(re.search(r'/s(\d+)/', sd_dir + '/').group(1))
        rows = []
        for model_dir in sorted(glob.glob(os.path.join(sd_dir, '*/'))):
            model_full = os.path.basename(model_dir.rstrip('/'))
            model = re.sub(r'_s\d+$', '', model_full)
            for f in sorted(glob.glob(os.path.join(model_dir, 'key_*.mid'))):
                m = re.match(r'key_([A-G]#?)(?:_t(\d+))?\.mid$', os.path.basename(f))
                key = KEY_OF[m.group(1)]
                prompt = int(m.group(2) or 0)
                row = {'model': model, 'key': key, 'key_name': m.group(1),
                       'group': 'unseen' if key in UNSEEN else 'finetuned',
                       'prompt': prompt, 'file': os.path.relpath(f, args.midi_root),
                       'song': -1}
                row.update(score_file(f, key, n_prompt, args.beats_per_chord,
                                      args.n_gen_beats))
                rows.append(row)
        out = os.path.join(args.out_dir, f'prompt_mix_midi_s{seed}_{args.protocol}.csv')
        with open(out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        models = sorted({r['model'] for r in rows})
        print(f'seed {seed}: {len(rows)} files, {len(models)} models -> {out}')


if __name__ == '__main__':
    main()
