"""
Separate "the model did not learn" from "generation is broken".

Rule-following is measured by free-running the model for 16-23 beats. A model
can have near-zero teacher-forced loss and still free-run badly, and the two
failures need completely different fixes, so measure both on the same
checkpoint and the same sequences:

  teacher-forced : feed the GROUND-TRUTH beats, read the argmax prediction for
                   each next beat's voice-0 pitch class. This is what the
                   training loss actually optimises.
  free-run       : the eval's own path -- prompt, then sample autoregressively.

  tf high, fr low  -> exposure bias or a bug in the sampling path. More
                      pretraining will not help.
  tf low,  fr low  -> the model genuinely did not learn the rule, and a
                      train_loss near zero means the loss is dominated by
                      slots that are not the bass pitch.

Usage
-----
  python -m midi_adapter.diagnose_cp_bass --base_ckpt <ckpt> --model_size 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.generate_synthetic_bass import generate_song, _preprocess_pm, OFFSETS
from midi_adapter.evaluate_cp_bass import (
    load_model, _extract_pc, _gen_pcs, PRETRAIN_ONLY, BOTH_SEEN, UNSEEN,
)


@torch.no_grad()
def teacher_forced_pc_acc(base, key: int, T: int, device):
    """Argmax voice-0 pitch class at each beat, given ground-truth context.

    Returns (accuracy, predicted pcs, expected pcs). Beat 0 is skipped: with
    h = cat([sos, h[:-1]]) the model has no information there.
    """
    pm, _ = generate_song(n_bars=max(1, -(-T // 4)), key=key, quality='maj')
    data, _ = _preprocess_pm(pm, T)
    x = base.preprocess(data.unsqueeze(0).to(device),
                        torch.zeros(1, dtype=torch.long, device=device))

    # forward returns (B*seq_len, subseq_len, vocab) -- flattened over time,
    # not (B, T, subseq_len, vocab).
    logits = base(x)
    pred_tok = logits[:, 1].argmax(-1)           # slot 1 = voice-0 pitch token
    pred_pc  = [(int(t) % 128) % 12 for t in pred_tok]
    exp_pc   = [(key + OFFSETS[t % 4]) % 12 for t in range(T)]

    hits = [p == e for p, e in zip(pred_pc[1:], exp_pc[1:])]
    return float(np.mean(hits)), pred_pc, exp_pc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt', required=True)
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--n_beats', type=int, default=16)
    p.add_argument('--n_prompt_beats', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = load_model(args.base_ckpt, args.model_size, device)

    keys = PRETRAIN_ONLY + BOTH_SEEN + UNSEEN
    print(f'\n{"key":>4}  {"teacher-forced":>15}  {"free-run":>9}   verdict')
    print('-' * 62)
    tf_all, fr_all = [], []
    for key in keys:
        tf, pred, exp = teacher_forced_pc_acc(base, key, args.n_beats, device)
        fr_pcs = _gen_pcs(base, key, args.n_beats, device, 0.0, args.n_prompt_beats)
        fr_exp = [(key + OFFSETS[(t + args.n_prompt_beats) % 4]) % 12
                  for t in range(len(fr_pcs))]
        fr = float(np.mean([a == b for a, b in zip(fr_pcs, fr_exp)]))
        tf_all.append(tf); fr_all.append(fr)
        tag = ('seen' if key in PRETRAIN_ONLY + BOTH_SEEN else 'unseen')
        print(f'{key:>4}  {tf:>15.3f}  {fr:>9.3f}   {tag}')
        if key == keys[0]:
            print(f'      expected  : {exp[:12]}')
            print(f'      tf pred   : {pred[:12]}')
            print(f'      free-run  : {fr_pcs[:12]}')
    print('-' * 62)
    print(f'{"mean":>4}  {np.mean(tf_all):>15.3f}  {np.mean(fr_all):>9.3f}')
    print()
    if np.mean(tf_all) > 0.9 and np.mean(fr_all) < 0.5:
        print('  -> teacher-forced works, free-run does not: exposure bias or a')
        print('     bug in the sampling path. More pretraining will not help.')
    elif np.mean(tf_all) < 0.5:
        print('  -> the model cannot predict the next bass pitch even WITH the')
        print('     ground truth in context. A low train_loss then means the loss')
        print('     is dominated by slots other than the bass pitch.')


if __name__ == '__main__':
    main()
