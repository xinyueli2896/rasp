"""Why does the in-training probe disagree with the results table?

On one latent-rule checkpoint the probe reported unseen_acc 0.618 while the
table reported 0.329 for the same weights. The two paths share their scoring
(same downbeat positions, same expected-root formula, same validity test) and
both call global_sampling, so the only difference is where the prompt comes
from:

  probe  real windows from the unseen .pt, first n_prompt_beats as the prompt
  table  a synthetic song built by _prompt_from_key for a named key

This runs both on the same checkpoint and reports them side by side, plus a
third row that scores the real windows' own continuation, which bounds what is
achievable on those windows at all.

Usage
-----
  python -m midi_adapter.diagnose_probe_gap \\
      --base_ckpt    ckpt/cp_bass_pre_bar_s1/....step004000.ckpt \\
      --adapter_ckpt checkpoints_bar/cp_yy_frozen_s1/....ckpt \\
      --unseen_data  data/bass_unseen_bar.pt \\
      --beats_per_chord 4 --max_polyphony 2 \\
      --n_prompt_beats 4 --n_gen_beats 28
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.evaluate_cp_bass import UNSEEN, _expected_pc, _scored_positions
from midi_adapter.evaluate_cp_yinyang import load_model
from midi_adapter.infer_cp_bass import _prompt_from_key
from midi_adapter.train_cp_yinyang import BassFramedDataset


def _pc(row: torch.Tensor, tokenizer) -> int | None:
    """Pitch class of one preprocessed beat, or None when it holds no note."""
    prog, pitch_dur = int(row[0]), int(row[1])
    if prog >= 128 or pitch_dur < 128:
        return None
    return (pitch_dur % 128) % 12


@torch.no_grad()
def score(model, prompt, key, n_prompt, n_gen, bpc):
    """Generate from one prompt and score its downbeats. Returns (hits, total)."""
    total = n_prompt + n_gen
    sampled = model.global_sampling(prompt, max_seq_len=total, temperature=0)
    hits = n = 0
    for t in _scored_positions(n_prompt, total, bpc):
        got = _pc(sampled[t][0], model.base.tokenizer)
        n += 1
        hits += int(got is not None and got == _expected_pc(key, t, bpc))
    return hits, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base_ckpt', required=True)
    p.add_argument('--adapter_ckpt', required=True)
    p.add_argument('--unseen_data', required=True)
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip', type=int, default=1)
    p.add_argument('--rule_mode', type=str, default='compiled')
    p.add_argument('--rule_view', type=str, default='root_next')
    p.add_argument('--rule_input', type=str, default='tokens')
    p.add_argument('--lora_rank', type=int, default=0)
    p.add_argument('--beats_per_chord', type=int, default=4)
    p.add_argument('--rule_key_select', type=str, default='first',
                   choices=['phase0', 'first'])
    p.add_argument('--max_polyphony', type=int, default=2)
    p.add_argument('--n_prompt_beats', type=int, default=4)
    p.add_argument('--n_gen_beats', type=int, default=28)
    p.add_argument('--n_windows', type=int, default=40)
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    dev = torch.device(args.device if args.device else
                       ('cuda' if torch.cuda.is_available() else 'cpu'))
    total = args.n_prompt_beats + args.n_gen_beats
    bpc = args.beats_per_chord

    model = load_model(args.base_ckpt, args.adapter_ckpt, args.model_size,
                       args.adapter_rank, args.n_skip, False, False,
                       'embedding', args.rule_mode, dev,
                       rule_view=args.rule_view, lora_rank=args.lora_rank,
                       rule_input=args.rule_input,
                       rule_subbeats_per_chord=args.beats_per_chord,
                       rule_key_select=args.rule_key_select)
    model.eval()

    print('=' * 68)
    print(f'protocol: {args.n_prompt_beats}-beat prompt, {args.n_gen_beats} '
          f'generated, downbeats scored (beats_per_chord={bpc})')
    print('=' * 68)

    # --- A: real windows, exactly what the probe prompts with ---------------
    ds = BassFramedDataset(args.unseen_data, total, 8, split='val',
                           sample_step=4 * bpc, repeat=True)
    ds.preload({})
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    hits = n = 0
    per_key_real: dict[int, list] = {}
    seen = 0
    for x, pitch_shift in loader:
        x, pitch_shift = x.to(dev), pitch_shift.to(dev)
        proc = model.base.preprocess(x, pitch_shift)
        for b in range(proc.shape[0]):
            if seen >= args.n_windows:
                break
            key = int(proc[b, 0, 1] % 128 % 12)
            h, m = score(model, proc[b:b + 1, :args.n_prompt_beats],
                         key, args.n_prompt_beats, args.n_gen_beats, bpc)
            hits += h; n += m; seen += 1
            per_key_real.setdefault(key, []).append(h / max(m, 1))
        if seen >= args.n_windows:
            break
    print(f'\nA  real windows   ({seen} windows from {os.path.basename(args.unseen_data)})')
    print(f'     accuracy {hits / max(n, 1):.3f}   ({hits}/{n} downbeats)')
    print(f'     keys present: '
          + '  '.join(f'{k}:{np.mean(v):.3f}(n={len(v)})'
                      for k, v in sorted(per_key_real.items())))

    # --- B: synthetic prompts, exactly what the table prompts with ----------
    hits = n = 0
    per_key_syn = {}
    for key in UNSEEN:
        kh = kn = 0
        for _ in range(args.n_trials):
            prompt = _prompt_from_key(key, 0, dev, model.base,
                                      n_prompt_beats=args.n_prompt_beats,
                                      max_polyphony=args.max_polyphony,
                                      cycle_unit='bar' if bpc > 1 else 'beat')
            h, m = score(model, prompt, key,
                         args.n_prompt_beats, args.n_gen_beats, bpc)
            kh += h; kn += m
        per_key_syn[key] = kh / max(kn, 1)
        hits += kh; n += kn
    print(f'\nB  synthetic prompts ({args.n_trials} trials x {len(UNSEEN)} keys)')
    print(f'     accuracy {hits / max(n, 1):.3f}   ({hits}/{n} downbeats)')
    print(f'     per key: ' + '  '.join(f'{k}:{v:.3f}' for k, v in per_key_syn.items()))

    print('\n' + '-' * 68)
    print('If A >> B the two prompt sources are not equivalent and the table is')
    print('measuring something harder than the probe. Compare the key coverage')
    print('above: the probe averages whatever keys its windows happen to hold,')
    print('while the table weights every unseen key equally.')


if __name__ == '__main__':
    main()
