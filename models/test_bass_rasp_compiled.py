"""Tests for the complete Tracr compilation of the bass rule.

Run: python -m models.test_bass_rasp_compiled
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bass_rasp_compiled import (
    BassRaspCompiled, N_ROOTS, N_POS, KEY_BASE, ROOT_BASE, TOK_BASE, PHASE_BASE,
)
from models.bass_tracr_rule_model import BassTracrRuleModel
from rasp_program.sequence_rule import OFFSETS

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def seq(key, T):
    return [(key + OFFSETS[t % N_POS]) % N_ROOTS for t in range(T)]


def main():
    m = BassRaspCompiled()
    T = 24

    print('\n1. exactness: root == (key + OFFSETS[phase]) % 12, every key, every position')
    worst = 1.0
    for key in range(N_ROOTS):
        idx = torch.tensor([seq(key, T)])
        logits, h = m(idx, return_hidden=True)
        want = torch.tensor([seq(key, T)])
        ok = bool((logits.argmax(-1) == want).all())
        worst = min(worst, float(logits.max(-1).values.min()))
        if not ok:
            check(f'key={key}', False, f'got {logits.argmax(-1)[0].tolist()}')
            return
    check('all 12 keys exact', True, f'min root activation = {worst:.3f}')
    check('root is a clean one-hot', abs(worst - 1.0) < 1e-5)

    print('\n2. the MLP does the arithmetic, not Python')
    # Feed the key subspace directly, bypassing tokens entirely: if the
    # arithmetic were done outside the network this could not work.
    for key in (0, 6, 8):
        x = torch.zeros(1, T, m.d_model)
        x[0, :, KEY_BASE + key] = 1.0
        x[0, torch.arange(T), PHASE_BASE + (torch.arange(T) % N_POS)] = 1.0
        out = m.mlp(x)[0, :, ROOT_BASE:ROOT_BASE + N_ROOTS].argmax(-1)
        check(f'mlp alone computes roots for key={key}', out.tolist() == seq(key, T))

    print('\n3. reads the OBSERVED tokens (BassTracrRuleModel does not)')
    key = 3
    idx = torch.tensor([seq(key, T)])
    # Corrupt a phase-0 slot: the retrieved key must change.
    bad0 = idx.clone(); bad0[0, 4] = (key + 1) % 12
    # Corrupt a non-phase-0 slot: retrieval only reads phase-0, so no change.
    bad1 = idx.clone(); bad1[0, 5] = (key + 1) % 12
    _, h  = m(idx,  return_hidden=True)
    _, h0 = m(bad0, return_hidden=True)
    _, h1 = m(bad1, return_hidden=True)
    k  = h[0, :, KEY_BASE:KEY_BASE + N_ROOTS]
    k0 = h0[0, :, KEY_BASE:KEY_BASE + N_ROOTS]
    k1 = h1[0, :, KEY_BASE:KEY_BASE + N_ROOTS]
    check('corrupting a phase-0 token moves the retrieved key',
          float((k - k0).abs().max()) > 0.1, f'delta={float((k - k0).abs().max()):.3f}')
    check('corrupting a non-phase-0 token does not',
          float((k - k1).abs().max()) < 1e-6, f'delta={float((k - k1).abs().max()):.3g}')
    # Token subspace must echo the input, not the rule.
    _, hb = m(bad1, return_hidden=True)
    echoed = hb[0, :, TOK_BASE:TOK_BASE + N_ROOTS].argmax(-1).tolist()
    check('token dims echo the OBSERVED input', echoed == bad1[0].tolist())
    _, hold = BassTracrRuleModel(rule_mode='current')(bad1, return_hidden=True)
    old_tok = hold[0, :, 0:N_ROOTS].argmax(-1).tolist()
    check('BassTracrRuleModel instead asserts the rule', old_tok != bad1[0].tolist(),
          f'old={old_tok[:6]} input={bad1[0].tolist()[:6]}')

    print('\n4. causality: position t never depends on tokens > t')
    idx = torch.tensor([seq(5, T)])
    _, h_ref = m(idx, return_hidden=True)
    bad = idx.clone(); bad[0, T // 2:] = 11
    _, h_bad = m(bad, return_hidden=True)
    pre = float((h_ref[0, :T // 2] - h_bad[0, :T // 2]).abs().max())
    check('prefix unchanged when the suffix is corrupted', pre < 1e-6, f'delta={pre:.3g}')

    print('\n5. frozen but differentiable')
    check('zero trainable parameters', sum(p.numel() for p in m.parameters()) == 0)
    enc = torch.randn(1, T, N_ROOTS, requires_grad=True)
    x = F.pad(enc, (0, m.d_model - N_ROOTS)) \
        + m.W_pos[m.phase_at(T, 'cpu')].unsqueeze(0)
    out = m.mlp(m.attention(x))
    out[:, :, ROOT_BASE:].sum().backward()
    g = float(enc.grad.norm())
    check('gradient reaches a float proxy through head AND mlp', g > 0, f'norm={g:.3f}')

    print('\n6. BOS convention: a query with no selected key falls back, not averages')
    # phase_at with subbeats_per_chord=2 puts phase 0 only at t in {0,1}.
    m2 = BassRaspCompiled(subbeats_per_chord=2)
    ph = m2.phase_at(8, 'cpu').tolist()
    idx2 = torch.tensor([[ (7 + OFFSETS[p]) % 12 for p in ph ]])
    _, h2 = m2(idx2, return_hidden=True)
    check('retrieval still exact at coarser slot width',
          h2[0, :, KEY_BASE:KEY_BASE + N_ROOTS].argmax(-1).tolist() == [7] * 8,
          f'phase={ph}')
    # A stream with NO phase-0 key at all: BOS must absorb it (all-zero key).
    x = torch.zeros(1, 4, m2.d_model)
    x[0, :, PHASE_BASE + 1] = 1.0                 # every position phase 1
    x[0, :, TOK_BASE + 3]   = 1.0
    kk = m2.attention(x)[0, :, KEY_BASE:KEY_BASE + N_ROOTS]
    check('no match -> key subspace stays ~empty', float(kk.abs().max()) < 1e-3,
          f'max={float(kk.abs().max()):.3g}')

    print('\n7. no layer norm, one attention + one mlp')
    check('no LayerNorm modules', not any(isinstance(mod, torch.nn.LayerNorm)
                                          for mod in m.modules()))
    check('d_model = 12*3 + 4 = 40', m.d_model == 40)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
