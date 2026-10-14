"""
Why does the frozen-AR adapter free-run far below what it was trained to do?

Separates three numbers that the single rule-following accuracy conflates:

  teacher-forced   the training computation exactly -- every position gets its
                   adapter correction, inputs are ground truth. Upper bound on
                   what the trained adapter knows.

  free-run ALL     autoregressive, but the correction is applied to EVERY
                   position of the recomputed sequence, so the host's context
                   matches training. Isolates the cost of generating its own
                   inputs.

  free-run LAST    autoregressive with the correction applied only at the last
                   position -- what CPYinyangTransformer.global_sampling does
                   today. Isolates the cost of the train/inference mismatch.

The mismatch is real: the training forward (cp_yinyang.py:473) runs

    h = h + yinyang_attn[k](h, rule_h, sub_offset=0)        # ALL positions

while global_sampling (cp_yinyang.py:545) runs

    correction = yinyang_attn[k](h_out[:, -1:, :], rule_h, sub_offset=i)
    h_out = cat([h_out[:, :-1, :], h_out[:, -1:, :] + correction], dim=1)

and each step recomputes all global layers from `h`, which accumulates only
local-encoded beats -- never block outputs. So an earlier position's correction
is neither persisted nor recomputed, and the host's self-attention at position i
reads the UNCORRECTED base representation of every earlier beat. In the frozen
stem the correction is the only thing that changes the host's behaviour, so
dropping it from the context leaves position i attending to something that looks
like the plain pretrain base -- which scores 0.000 on unseen keys.

A large ALL-vs-LAST gap means the deployed sampler, not the adapter, is what the
Unseen accuracy is measuring.

Usage
-----
  python -m midi_adapter.debug_frozen_adapter \\
      --base_ckpt    ckpt/cp_bass_pre_A_aligned/....ckpt \\
      --adapter_ckpt checkpoints_cp/cp_yy_frozen/....ckpt \\
      --n_prompt_beats 1 --n_gen_beats 23
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.evaluate_cp_bass import CATEGORIES, ALL_SEEN, UNSEEN, _expected_pc
from midi_adapter.evaluate_cp_yinyang import load_model
from midi_adapter.infer_cp_bass import _prompt_from_key


# ---------------------------------------------------------------------------
# rule_hidden, exactly as global_sampling builds it
# ---------------------------------------------------------------------------

def _rule_hidden_for_sampling(model: CPYinyangTransformer, x: torch.Tensor,
                              max_seq_len: int) -> torch.Tensor | None:
    if model.bidirectional:
        return None
    pc_0 = (x[:, 0, 1] % 128 % 12).clamp(0, 11)
    return model.rule_model.build_rule_hidden_analytic(pc_0, max_seq_len, x.device)


# ---------------------------------------------------------------------------
# Free-running generation, with the correction site under our control
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model: CPYinyangTransformer, x: torch.Tensor, max_seq_len: int,
             correct_all: bool) -> list[torch.Tensor]:
    """global_sampling, but `correct_all` chooses who gets the correction.

    correct_all=False reproduces the shipped sampler exactly.
    correct_all=True  reproduces the training forward's correction pattern.
    """
    base = model.base
    batch_size, seed_len, _ = x.shape
    rule_hidden = _rule_hidden_for_sampling(model, x, max_seq_len)

    h_seed, _ = base.local_encode(x)
    h_seed = h_seed.view(batch_size, seed_len, base.hidden_size)
    sos = base.global_sos.view(1, 1, -1).expand(batch_size, 1, -1)
    h = torch.cat([sos, h_seed], dim=1)

    y = [x[:, i, :] for i in range(seed_len)]

    for i in range(seed_len, max_seq_len):
        attn_mask = base.buffered_future_mask(h)
        sinusoidal_pos = base.model.embed_positions(h.shape[:-1], 0)[None, None, :, :]

        h_out = h
        for j, layer in enumerate(base.model.layer):
            h_out = layer(h_out, attention_mask=attn_mask,
                          sinusoidal_pos=sinusoidal_pos)[0]
            if (j + 1) % model.n_skip != 0:
                continue
            k = (j + 1) // model.n_skip - 1
            rule_h = (model.ar_to_rule[k](h_out) if model.bidirectional
                      else rule_hidden)
            if correct_all:
                # Whole sequence, absolute beat 0 at index 0 -- identical to
                # the training forward.
                h_out = h_out + model.yinyang_attn[k](h_out, rule_h, sub_offset=0)
            else:
                corr = model.yinyang_attn[k](h_out[:, -1:, :], rule_h, sub_offset=i)
                h_out = torch.cat([h_out[:, :-1, :], h_out[:, -1:, :] + corr], dim=1)

        y_next = base.local_sampling(h_out[:, -1], temperature=0)
        y.append(y_next)
        h_enc = base.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
        h = torch.cat([h, h_enc], dim=1)

    return y


# ---------------------------------------------------------------------------
# The three measurements
# ---------------------------------------------------------------------------

def _pc(tok_row: torch.Tensor) -> int:
    """Pitch class from a preprocessed CP beat: slot 1 = pitch + (dur+1)*128."""
    return int(tok_row[1] % 128 % 12)


@torch.no_grad()
def acc_teacher_forced(model, key, n_prompt, n_gen, device) -> float:
    """The training computation: ground-truth inputs, all positions corrected."""
    total = n_prompt + n_gen
    x = _prompt_from_key(key, n_prompt_bars=0, device=device,
                         base=model.base, n_prompt_beats=total)
    B, T, S = x.shape
    # local_encode flattens (B, T, subseq) -> (B*T, subseq), so local_decode
    # returns (B*T, subseq, vocab) rather than (B, T, subseq, vocab).
    logits = model(x)
    assert logits.shape[:2] == (B * T, S), \
        f'unexpected logits shape {tuple(logits.shape)}; expected ({B*T}, {S}, vocab)'
    pred = logits.argmax(-1).view(B, T, S)
    hits = [int(_pc(pred[0, t]) == _expected_pc(key, t))
            for t in range(n_prompt, total)]
    return float(np.mean(hits))


@torch.no_grad()
def acc_free_run(model, key, n_prompt, n_gen, device, correct_all) -> float:
    total = n_prompt + n_gen
    prompt = _prompt_from_key(key, n_prompt_bars=0, device=device,
                              base=model.base, n_prompt_beats=n_prompt)
    y = generate(model, prompt, total, correct_all=correct_all)
    hits = [int(_pc(y[t][0]) == _expected_pc(key, t))
            for t in range(n_prompt, total)]
    return float(np.mean(hits))


def main():
    p = argparse.ArgumentParser(
        description='Separate adapter quality from the sampler mismatch.')
    p.add_argument('--base_ckpt',    required=True)
    p.add_argument('--adapter_ckpt', required=True)
    p.add_argument('--model_size',     type=int, default=1)
    p.add_argument('--adapter_rank',   type=int, default=256)
    p.add_argument('--n_skip',         type=int, default=4)
    p.add_argument('--rule_mode',      type=str, default='compiled')
    p.add_argument('--rule_view',      type=str, default='root_next')
    p.add_argument('--lora_rank',      type=int, default=0)
    p.add_argument('--rule_input',     type=str, default='tokens')
    p.add_argument('--rule_enc_type',  type=str, default='mlp')
    p.add_argument('--rule_enc_hidden',type=int, default=64)
    p.add_argument('--n_prompt_beats', type=int, default=1)
    p.add_argument('--n_gen_beats',    type=int, default=23)
    p.add_argument('--device',         type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = load_model(args.base_ckpt, args.adapter_ckpt, args.model_size,
                       args.adapter_rank, args.n_skip, False, False,
                       'embedding', args.rule_mode, device,
                       rule_view       = args.rule_view,
                       lora_rank       = args.lora_rank,
                       rule_input      = args.rule_input,
                       rule_enc_type   = args.rule_enc_type,
                       rule_enc_hidden = args.rule_enc_hidden)
    model.eval()

    rows = list(CATEGORIES) + [('Overall', ALL_SEEN + UNSEEN, 'all 12 keys')]
    cols = [('teacher-forced', lambda k: acc_teacher_forced(
                model, k, args.n_prompt_beats, args.n_gen_beats, device)),
            ('free-run ALL',   lambda k: acc_free_run(
                model, k, args.n_prompt_beats, args.n_gen_beats, device, True)),
            ('free-run LAST',  lambda k: acc_free_run(
                model, k, args.n_prompt_beats, args.n_gen_beats, device, False))]

    print('=' * 78)
    print(f'FROZEN-ADAPTER DIAGNOSIS  ({args.n_prompt_beats}-beat prompt, '
          f'{args.n_gen_beats} generated, greedy)')
    print(f'  base    {args.base_ckpt}')
    print(f'  adapter {args.adapter_ckpt}')
    print('=' * 78)
    print('  free-run LAST = the shipped sampler.  free-run ALL = corrections at')
    print('  every position, matching the training forward.')
    print()
    hdr = f'{"Key group":<16} {"Keys":<22}' + ''.join(f' {c:>15}' for c, _ in cols)
    print(hdr); print('-' * len(hdr))

    res = {}
    for lbl, keys, kstr in rows:
        line = f'{lbl:<16} {kstr:<22}'
        for cname, fn in cols:
            v = float(np.mean([fn(k) for k in keys]))
            res[(lbl, cname)] = v
            line += f' {v:>15.3f}'
        print(line)
    print('=' * len(hdr))

    gap = res[('Unseen', 'free-run ALL')] - res[('Unseen', 'free-run LAST')]
    print()
    print(f'Unseen: ALL - LAST = {gap:+.3f}')
    if gap > 0.05:
        print('  The sampler mismatch accounts for most of the shortfall. Fix')
        print('  global_sampling to correct every position and re-evaluate --')
        print('  no retraining needed, the weights are unchanged.')
    elif res[('Unseen', 'teacher-forced')] - res[('Unseen', 'free-run ALL')] > 0.05:
        print('  The sampler is fine; the loss is in generating its own inputs.')
        print('  Exposure bias, not a plumbing bug.')
    else:
        print('  Neither. The adapter genuinely did not learn to route the rule')
        print('  signal on unseen keys -- the teacher-forced number is the')
        print('  ceiling, and it is already low.')


if __name__ == '__main__':
    main()
