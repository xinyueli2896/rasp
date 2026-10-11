#!/usr/bin/env python3
"""Checks on the Tracr-compiled integer rule model and its PyTorch lift.

Four properties are claimed; all four are things the PREVIOUS rule model did
not have, so none of them should be taken on trust:

  1. FIDELITY      the lift matches real Tracr on the full residual stream,
                   not merely on the decoded argmax
  2. NO FIRST CYCLE correct from T = 1, which is what generation from a
                   one-token prompt needs
  3. CONTEXT-FREE  the output depends on (token[0], position) and nothing else
  4. GRADIENT FLOW frozen to the optimizer, differentiable to the input

Sections 1 and 3b need `tracr`+`jax`; everything else runs from the cached
.pt and is skipped cleanly if the cache is missing.

  python -m models.test_integer_rule_model
  python -m models.test_integer_rule_model --max_seq_len 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.integer_rule_model import IntegerRuleModel, cache_path
from models.tracr_to_torch import TracrTorch, verify_against_tracr
from rasp_program.next_token_rule import (
    OFFSETS, VOCAB_SIZE, next_token, sequence,
)

FAILS: list[str] = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_seq_len', type=int, default=32)
    p.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    p.add_argument('--seed_mode', type=str, default='first',
                   choices=['first', 'every_cycle'])
    a = p.parse_args()
    V, T = a.vocab_size, a.max_seq_len

    try:
        import tracr  # noqa: F401
        import jax    # noqa: F401
        have_tracr = True
    except ImportError:
        have_tracr = False

    path = cache_path(T, V, a.seed_mode)
    if os.path.exists(path):
        m = IntegerRuleModel.load(path)
        print(f'loaded {path}')
    elif have_tracr:
        m = IntegerRuleModel.build(T, V, seed_mode=a.seed_mode)
        print('compiled fresh (no cache)')
    else:
        print(f'no cache at {path} and no tracr installed; nothing to test.')
        return 1
    print(f'd_model={m.d_model} max_seq_len={m.max_seq_len} '
          f'seed_mode={m.seed_mode} V={m.vocab_size}\n')

    # -- 1. fidelity of the lift ------------------------------------------
    print('1. the PyTorch lift reproduces Tracr')
    if have_tracr:
        from rasp_program.next_token_rule import compile_program
        asm = compile_program(max_seq_len=T, vocab_size=V, seed_mode=a.seed_mode)
        lifted = TracrTorch.from_assembled(asm, vocab_size=V,
                                           output_name='next_tok')
        seqs = [sequence([x], L)[0].tolist()
                for x in range(V) for L in (1, 2, 3, 5, 8, min(T, 16))]
        worst, _ = verify_against_tracr(lifted, asm, seqs)
        check('full residual stream matches Tracr', worst < 1e-5,
              f'max|diff| = {worst:.2e} over {len(seqs)} sequences')
        # Agreement on the stream is the claim that matters: the adapter reads
        # the stream, not the argmax, and two models can decode alike while
        # differing everywhere the adapter looks.
    else:
        print('  SKIP  (tracr not installed)')

    # -- 2. no first cycle needed -----------------------------------------
    print('\n2. correct without a first cycle in context')
    bad, lens = [], [1, 2, 3, 4, 5, 8, min(T, 16)]
    for x in range(V):
        for L in lens:
            seq = torch.as_tensor(sequence([x], L))
            _, lg = m.from_tokens(seq)
            if not (lg.argmax(-1)[0].numpy() == next_token([x], L)[0]).all():
                bad.append((x, L))
    check(f'every starter x every length in {lens}', not bad,
          f'{V * len(lens)} cases' if not bad else f'failed {bad[:6]}')

    one = [(int(m.from_tokens(torch.tensor([[x]]))[1].argmax(-1)[0, 0]),
            int(next_token([x], 1)[0, 0])) for x in range(V)]
    check('T=1 (a one-token prompt) is already correct',
          all(g == e for g, e in one),
          f'got {[g for g, _ in one]}')

    # The old retrieval model was wrong exactly here. Show the three positions.
    seq8 = torch.as_tensor(sequence([0], 8))
    got8 = m.from_tokens(seq8)[1].argmax(-1)[0].tolist()
    check('q = 0,1,2 correct (the old model\'s "honest uncertainty" hole)',
          got8[:3] == next_token([0], 8)[0][:3].tolist(),
          f'got {got8[:3]}, expect {next_token([0], 8)[0][:3].tolist()}')

    # -- 3. context independence ------------------------------------------
    print('\n3. output depends on (token[0], position) and nothing else')
    base = torch.as_tensor(sequence([0], 12))
    h0, l0 = m.from_tokens(base)
    rng = np.random.default_rng(0)
    worst = {nm: 0.0 for nm in ('next_tok', 'seed', 'next_phase',
                                'indices', 'one', 'tokens')}
    worst_prog = 0.0
    for _ in range(25):
        corrupt = base.clone()
        pos = int(rng.integers(1, 12))                 # never position 0
        corrupt[0, pos] = int(rng.integers(0, V))
        hc, _ = m.from_tokens(corrupt)
        for nm in worst:
            d = m.subspace(nm)
            worst[nm] = max(worst[nm], float((hc - h0)[..., d].abs().max()))
        worst_prog = max(worst_prog,
                         float((hc - h0)[..., m.program_dims].abs().max()))
    # The COMPUTED variables are what the claim is about. The residue is the
    # softmax tail: non-matching keys get weight ~1e-9, not exactly 0.
    #
    # This is the knob, not an accident. 'first' reads position 0 only, so it
    # is context-free and one position carries the whole signal. 'every_cycle'
    # averages over k % 4 == 0, so T/4 positions VOTE -- more leverage for a
    # learned encoder, at the price of being corruptible by the context it
    # reads. Assert whichever was built.
    if a.seed_mode == 'first':
        check('corrupting a later token leaves the COMPUTED subspaces alone',
              worst_prog < 1e-6,
              '  '.join(f'{nm}={worst[nm]:.1e}' for nm in
                        ('next_tok', 'seed', 'next_phase', 'indices', 'one')))
    else:
        check('every_cycle IS context-dependent (that is the trade)',
              worst_prog > 0.1,
              f'seed moves {worst["seed"]:.2f} when a cycle-start token changes')
    # ...and the one thing that does move is the raw token echo, which is
    # exactly why program_hidden() drops it.
    check('the only subspace that moves is the `tokens` echo',
          worst['tokens'] > 0.5, f"tokens={worst['tokens']:.3f}")
    check('program_hidden() excludes the echo subspaces',
          m.program_dims.numel() == m.d_model - len(m.subspace('tokens'))
          - len(m.subspace('indices')) - len(m.subspace('one')),
          f'{m.program_dims.numel()} of {m.d_model} dims kept')

    # Off the simplex corners too, not just at them: a learned encoder emits
    # soft vectors, so flatness only at one-hots would be no use.
    drift = 0.0
    for al in (0.1, 0.25, 0.5, 0.75, 0.9) if a.seed_mode == 'first' else ():
        p = torch.zeros(1, 12, V)
        p[0, torch.arange(12), base[0]] = 1.0
        p[0, 5] = 0.0
        p[0, 5, int(base[0, 5])] = 1.0 - al
        p[0, 5, int((base[0, 5] + 3) % V)] = al
        h, _ = m(p)
        drift = max(drift, float((h - h0)[..., m.program_dims].abs().max()))
    if a.seed_mode == 'first':
        check('still flat for SOFT corruption of a later token (not just one-hots)',
              drift < 1e-6, f'max drift = {drift:.2e}')

    h_other, _ = m.from_tokens(torch.as_tensor(sequence([7], 12)))
    check('changing token[0] DOES change the hidden state',
          float((h_other - h0).abs().max()) > 0.5,
          f'max|diff| = {float((h_other - h0).abs().max()):.3f}')

    if a.seed_mode == 'first':
        # Causality is argued from the program text ("k == 0 is always in the
        # causal prefix"); assert it numerically instead.
        hc, _ = m.from_tokens(base, causal=False)
        check('causal mask is a no-op for this program',
              float((hc - h0).abs().max()) < 1e-6,
              f'max|diff| = {float((hc - h0).abs().max()):.2e}')

    # -- 4. shift-by-1 alignment ------------------------------------------
    print('\n4. shift-by-1 alignment with the AR model')
    L = 8
    seq = sequence([3], L)[0]
    _, lg = m.from_tokens(torch.as_tensor(seq).unsqueeze(0))
    pred = lg.argmax(-1)[0].numpy()
    check('hidden[:, q] predicts token q+1, not token q',
          (pred[:-1] == seq[1:]).all(),
          f'pred[:-1]={pred[:-1].tolist()} vs seq[1:]={seq[1:].tolist()}')
    check('the last column extrapolates one past the input',
          pred[-1] == next_token([3], L)[0, -1],
          f'{pred[-1]} == {next_token([3], L)[0, -1]}')

    # -- 5. frozen but differentiable -------------------------------------
    print('\n5. frozen to the optimizer, differentiable to the input')
    check('parameters() is empty', len(list(m.parameters())) == 0)
    check('named_parameters() is empty', len(list(m.named_parameters())) == 0)
    n_buf = len(list(m.buffers()))
    check('weights are buffers (move with .to(), save in state_dict)', n_buf > 0,
          f'{n_buf} buffers')
    check('no buffer requires grad',
          all(not b.requires_grad for b in m.buffers()))

    probs = torch.zeros(1, 8, V)
    probs[0, torch.arange(8), torch.as_tensor(sequence([0], 8)[0])] = 1.0
    probs.requires_grad_(True)
    h, lg = m(probs)
    lg.sum().backward()
    check('gradient reaches the input distribution',
          probs.grad is not None and float(probs.grad.abs().sum()) > 0,
          f'|grad| = {float(probs.grad.abs().sum()):.4f}')

    # The input is a DISTRIBUTION, so the meaningful derivative is the one
    # along the simplex (sum stays 1). The raw gradient at q>0 is nonzero but
    # points purely along the all-ones "total mass" direction, which no
    # normalised encoder can move. Project it out before judging.
    g   = probs.grad[0]
    tan = g - g.mean(-1, keepdim=True)
    per_pos = tan.abs().sum(-1)
    if a.seed_mode == 'first':
        check('on the simplex, only position 0 has gradient (only the seed matters)',
              float(per_pos[0]) > 1.0 and float(per_pos[1:].max()) < 1e-4,
              f'tangent: pos0={float(per_pos[0]):.2f}  '
              f'rest_max={float(per_pos[1:].max()):.1e}')
    else:
        # k % 4 == 0 -> positions 0, 4, 8 ... share the signal. This is the
        # redundancy that made the chord experiment's phase-0 vote work.
        votes = [q for q in range(8) if float(per_pos[q]) > 1e-3]
        check('every_cycle spreads gradient over the cycle starts (a vote)',
              votes == [0, 4], f'positions with gradient: {votes}')
    check('the residual gradient at q>0 is pure MASS direction',
          float(g[5].std()) < 1e-5, f'g[5] is constant at {float(g[5][0]):.3f}')

    # Consequence, and it is not academic: the chord experiment's encoder uses
    # a per-dimension sigmoid, whose output does NOT sum to 1. Unnormalised
    # mass leaks into the rule model through exactly that direction.
    base8, exp8 = sequence([0], 8)[0], next_token([0], 8)[0]
    broke_at = None
    for s in (1.2, 1.5, 2.0, 3.0):
        p = torch.zeros(1, 8, V)
        p[0, torch.arange(8), torch.as_tensor(base8)] = 1.0
        p[0, 5] *= s
        if not (m(p)[1].argmax(-1)[0].numpy() == exp8).all():
            broke_at = s
            break
    check('tolerates mild mass error but breaks by 2x -> the encoder must be '
          'NORMALISED (softmax), not per-dimension (sigmoid)',
          broke_at is not None and broke_at >= 1.5,
          f'first wrong prediction at mass x{broke_at}')

    # A soft input must agree with the hard one it is a relaxation of.
    hard = torch.zeros(1, 6, V)
    hard[0, torch.arange(6), torch.as_tensor(sequence([2], 6)[0])] = 1.0
    h_soft, _ = m(hard)
    h_hard, _ = m.from_tokens(torch.as_tensor(sequence([2], 6)))
    check('soft path on a one-hot == integer path',
          float((h_soft - h_hard).abs().max()) < 1e-6,
          f'max|diff| = {float((h_soft - h_hard).abs().max()):.2e}')

    # -- 6. named subspaces the adapter will slice ------------------------
    print('\n6. residual subspaces are addressable by name')
    for nm, width in (('next_tok', V), ('seed', V), ('next_phase', 4)):
        try:
            dims = m.subspace(nm)
            check(f'subspace {nm!r} has width {width}', len(dims) == width,
                  f'dims {dims[0]}..{dims[-1]}')
        except KeyError as e:
            check(f'subspace {nm!r} exists', False, str(e))

    seqt = torch.as_tensor(sequence([9], 8))
    h, _ = m.from_tokens(seqt)
    seed_slot = h[..., m.subspace('seed')].argmax(-1)[0]
    check('the seed subspace holds token[0] at every position',
          bool((seed_slot == 9).all()), f'{seed_slot.tolist()}')

    ph = h[..., m.subspace('next_phase')].argmax(-1)[0]
    check('the next_phase subspace holds (q+1) % 4',
          ph.tolist() == [(q + 1) % 4 for q in range(8)], f'{ph.tolist()}')

    print(f'\n{"all checks passed" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
