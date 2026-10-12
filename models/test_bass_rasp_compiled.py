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
    BassRaspCompiled, VIEWS, VIEW_WIDTH, view_dims,
    N_ROOTS, N_POS, TOK_BASE, KEY_BASE, PH_BASE, PHN_BASE, ROOT_BASE, D_MODEL,
)
from models.bass_tracr_rule_model import BassTracrRuleModel
from rasp_program.sequence_rule import OFFSETS

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def seq(key, T):
    return [(key + OFFSETS[t % N_POS]) % N_ROOTS for t in range(T)]


def main():
    T = 24
    m = BassRaspCompiled()           # shift=1 -> next root

    print('\n1. exactness: root_next[t] == token[t+1], every key, every position')
    worst = 1.0
    for key in range(N_ROOTS):
        idx = torch.tensor([seq(key, T)])
        logits = m(idx)
        got = logits.argmax(-1)[0, :T - 1].tolist()
        if got != idx[0, 1:].tolist():
            check(f'key={key}', False, f'got {got}')
            return 1
        worst = min(worst, float(logits.max(-1).values.min()))
    check('all 12 keys predict the next root exactly', True,
          f'min activation = {worst:.3f}')
    check('output is a clean one-hot', abs(worst - 1.0) < 1e-5)

    print('\n2. shift=0 gives the CURRENT root instead')
    m0 = BassRaspCompiled(shift=0)
    idx = torch.tensor([seq(6, T)])
    check('shift=0 reproduces the input', m0(idx).argmax(-1)[0].tolist() == seq(6, T))

    print('\n3. the MLP does the arithmetic, not Python')
    for key in (0, 6, 8):
        x = torch.zeros(1, T, D_MODEL)
        x[0, :, KEY_BASE + key] = 1.0
        ph = ((torch.arange(T) + 1) % N_POS)
        x[0, torch.arange(T), PHN_BASE + ph] = 1.0
        out = m.mlp(x)[0, :, ROOT_BASE:ROOT_BASE + N_ROOTS].argmax(-1).tolist()
        want = [(key + OFFSETS[p]) % N_ROOTS for p in ph.tolist()]
        check(f'mlp alone computes next roots for key={key}', out == want)

    print('\n4. two phases are distinct and both positional')
    ph  = m.phase_at(T, 'cpu', 0).tolist()
    phn = m.phase_at(T, 'cpu', m.shift).tolist()
    check('phase != phase_next', ph != phn, f'{ph[:6]} vs {phn[:6]}')
    check('phase_next is phase shifted by 1', phn[:-1] == ph[1:])

    print('\n5. reads the OBSERVED tokens (BassTracrRuleModel does not)')
    key = 3
    idx = torch.tensor([seq(key, T)])
    bad = idx.clone(); bad[0, 5] = (key + 1) % 12      # not a phase-0 slot
    _, hb = m(bad, return_hidden=True)
    echoed = hb[0, :, TOK_BASE:TOK_BASE + N_ROOTS].argmax(-1).tolist()
    check('token dims echo the observed input', echoed == bad[0].tolist())
    _, hold = BassTracrRuleModel(rule_mode='current')(bad, return_hidden=True)
    old = hold[0, :, 0:N_ROOTS].argmax(-1).tolist()
    check('BassTracrRuleModel instead asserts the rule', old != bad[0].tolist(),
          f'old[5]={old[5]} input[5]={bad[0, 5].item()}')

    print('\n6. causality: position t never depends on tokens > t')
    idx = torch.tensor([seq(5, T)])
    _, h_ref = m(idx, return_hidden=True)
    bad = idx.clone(); bad[0, T // 2:] = 11
    _, h_bad = m(bad, return_hidden=True)
    d = float((h_ref[0, :T // 2] - h_bad[0, :T // 2]).abs().max())
    check('prefix unchanged when the suffix is corrupted', d < 1e-6, f'delta={d:.3g}')

    print('\n7. frozen but differentiable')
    check('zero trainable parameters', sum(p.numel() for p in m.parameters()) == 0)
    enc = torch.randn(1, T, N_ROOTS, requires_grad=True)
    x = F.pad(enc, (0, D_MODEL - N_ROOTS)) \
        + m.W_phase[m.phase_at(T, 'cpu', 0)].unsqueeze(0) \
        + m.W_phase_next[m.phase_at(T, 'cpu', m.shift)].unsqueeze(0)
    m.mlp(m.attention(x))[:, :, ROOT_BASE:].sum().backward()
    g = float(enc.grad.norm())
    check('gradient reaches a float proxy through head AND mlp', g > 0, f'norm={g:.3f}')

    print('\n8. BOS: a query selecting nothing contributes nothing')
    x = torch.zeros(1, 4, D_MODEL)
    x[0, :, PH_BASE + 1] = 1.0            # no phase-0 slot anywhere
    x[0, :, TOK_BASE + 3] = 1.0
    kk = m.attention(x)[0, :, KEY_BASE:KEY_BASE + N_ROOTS]
    check('no match -> key subspace stays empty', float(kk.abs().max()) < 1e-3,
          f'max={float(kk.abs().max()):.3g}')

    print('\n9. the four rule_hidden views')
    idx = torch.tensor([seq(6, T)])
    expected = {'root_next': 12, 'key_phase': 16, 'computed': 32, 'full': 44}
    for view, width in expected.items():
        mv = BassRaspCompiled(view=view)
        rh = mv.rule_hidden(idx)
        check(f'{view:<10} width {width}',
              VIEW_WIDTH[view] == width and rh.shape[-1] == width
              and mv.rule_d_model == width)
    # content checks
    mv = BassRaspCompiled(view='root_next')
    check('root_next view IS the answer',
          mv.rule_hidden(idx)[0, :T - 1].argmax(-1).tolist() == idx[0, 1:].tolist())
    mv = BassRaspCompiled(view='key_phase')
    rh = mv.rule_hidden(idx)
    check('key_phase view carries the key', rh[0, :, :N_ROOTS].argmax(-1).tolist()
          == [6] * T)
    check('key_phase view carries phase_next',
          rh[0, :, N_ROOTS:].argmax(-1).tolist() == mv.phase_at(T, 'cpu', 1).tolist())
    check('key_phase view does NOT contain the answer',
          rh.shape[-1] == 16 and ROOT_BASE not in view_dims('key_phase'))
    check('full view contains the token echo', TOK_BASE in view_dims('full'))
    check('computed view excludes the token echo',
          TOK_BASE not in view_dims('computed'))

    print('\n10. no layer norm')
    check('no LayerNorm modules',
          not any(isinstance(mod, torch.nn.LayerNorm) for mod in m.modules()))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
