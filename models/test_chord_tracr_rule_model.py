"""Tests for the v6 chord rule model (root AND quality).

Run: python -m models.test_chord_tracr_rule_model
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chord_tracr_rule_model import (
    ChordTracrRuleModel, QUAL_PATTERNS, QUALITIES, N_ROOTS, N_POS, N_QUAL,
)
from rasp_program.sequence_rule import OFFSETS

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok  " if cond else "FAIL"} {name}' + (f'   {detail}' if detail else ''))


def main():
    print('layout')
    m2 = ChordTracrRuleModel(('major', 'harmonic minor'))
    check('period4, 2 modes -> d_model 34', m2.d_model == 34, f'{m2.d_model}')
    m3 = ChordTracrRuleModel(tuple(QUAL_PATTERNS))
    check('period4, 3 modes -> d_model 35', m3.d_model == 35, f'{m3.d_model}')
    mc = ChordTracrRuleModel(('major', 'harmonic minor'), rule_mode='current')
    check('current, 2 modes -> d_model 20', mc.d_model == 20, f'{mc.d_model}')
    check('current has no next-* blocks',
          mc.sl_next_root is None and mc.sl_next_qual is None)
    check('zero parameters', len(list(m2.parameters())) == 0)
    check('nothing trainable in the state dict',
          all(not v.requires_grad for v in m2.state_dict().values()))

    print('\nthe rule, against a hand-written reference')
    T = 12
    keys = torch.arange(N_ROOTS)
    for mi, mode_name in enumerate(m3.modes):
        mode = torch.full((N_ROOTS,), mi)
        roots, quals = m3.expected(keys, mode, T)
        want_r = [[(int(k) + OFFSETS[t % N_POS]) % 12 for t in range(T)]
                  for k in keys]
        want_q = [[QUALITIES.index(QUAL_PATTERNS[mode_name][t % N_POS])
                   for t in range(T)] for _ in keys]
        check(f'{mode_name}: roots over all 12 keys',
              roots.tolist() == want_r)
        check(f'{mode_name}: qualities over all 12 keys',
              quals.tolist() == want_q)

    print('\nthe point of v6: roots are mode-blind, qualities are not')
    key = torch.tensor([0, 0])
    both = torch.tensor([0, 1])          # major, harmonic minor, same key
    r, q = m2.expected(key, both, 8)
    check('same key, two modes -> identical roots',
          torch.equal(r[0], r[1]), f'{r[0].tolist()}')
    check('same key, two modes -> different qualities',
          not torch.equal(q[0], q[1]), f'{q[0].tolist()} vs {q[1].tolist()}')
    check('harmonic minor flips only bar 2 of the cycle',
          (q[0] != q[1]).tolist()[:4] == [True, True, False, True])

    print('\nrule_hidden blocks')
    B, T = 6, 16
    torch.manual_seed(0)
    pc0 = torch.randint(0, 12, (B,))
    mode = torch.randint(0, 2, (B,))
    h = m2.build_rule_hidden_analytic(pc0, mode, T)
    check('shape', tuple(h.shape) == (B, T, m2.d_model), str(tuple(h.shape)))
    check('every value is 0 or 1', bool(((h == 0) | (h == 1)).all()))
    for name, sl, width in (('root', m2.sl_root, N_ROOTS),
                            ('qual', m2.sl_qual, N_QUAL),
                            ('next_root', m2.sl_next_root, N_ROOTS),
                            ('next_qual', m2.sl_next_qual, N_QUAL),
                            ('phase', m2.sl_phase, N_POS),
                            ('mode', m2.sl_mode, 2)):
        blk = h[:, :, sl]
        check(f'{name}: one-hot at every position',
              bool((blk.sum(-1) == 1).all()) and blk.shape[-1] == width)

    roots, quals = m2.expected(pc0, mode, T)
    check('root block decodes to the rule root',
          torch.equal(h[:, :, m2.sl_root].argmax(-1), roots))
    check('qual block decodes to the rule quality',
          torch.equal(h[:, :, m2.sl_qual].argmax(-1), quals))
    t = torch.arange(T)
    check('phase block is t % 4',
          torch.equal(h[:, :, m2.sl_phase].argmax(-1),
                      (t % N_POS)[None].expand(B, -1)))
    check('mode block is constant along t and equals the mode',
          torch.equal(h[:, :, m2.sl_mode].argmax(-1),
                      mode[:, None].expand(-1, T)))

    print('\nnext-* is the current-* of the following position')
    check('next_root[t] == root[t+1]',
          torch.equal(h[:, :-1, m2.sl_next_root], h[:, 1:, m2.sl_root]))
    check('next_qual[t] == qual[t+1]',
          torch.equal(h[:, :-1, m2.sl_next_qual], h[:, 1:, m2.sl_qual]))
    # The last position has no t+1 in the window, so it is checked against the
    # rule directly rather than against a neighbour that is not there.
    nr, nq = m2.expected(pc0, mode, T + 1)
    check('next_* at the last position continues the cycle',
          torch.equal(h[:, -1, m2.sl_next_root].argmax(-1), nr[:, T])
          and torch.equal(h[:, -1, m2.sl_next_qual].argmax(-1), nq[:, T]))

    print('\nforward')
    pc = torch.stack([(pc0 + torch.tensor(OFFSETS)[t % N_POS]) % 12
                      for t in range(T)], dim=1)
    rl, ql = m2(pc, mode)
    check('period4 logits predict the NEXT chord',
          torch.equal(rl.argmax(-1), nr[:, 1:T + 1])
          and torch.equal(ql.argmax(-1), nq[:, 1:T + 1]))
    rl, ql, h2 = m2(pc, mode, return_hidden=True)
    check('forward hidden == build_rule_hidden_analytic', torch.equal(h, h2))
    rlc, qlc = mc(pc, mode)
    check("rule_mode='current' logits give the CURRENT chord",
          torch.equal(rlc.argmax(-1), roots) and torch.equal(qlc.argmax(-1), quals))

    print('\nthe key is read from position 0, as in v4/v5')
    shifted = pc.clone()
    shifted[:, 0] = (shifted[:, 0] + 3) % 12        # a different tonic
    rl2, _ = m2(shifted, mode)
    check('changing pc[:, 0] changes the whole prediction',
          not torch.equal(rl2.argmax(-1), rl.argmax(-1)))

    print('\nvalidation')
    for name, fn in (
        ('unknown mode', lambda: ChordTracrRuleModel(('lydian',))),
        ('empty modes', lambda: ChordTracrRuleModel(())),
        ('bad rule_mode', lambda: ChordTracrRuleModel(rule_mode='period8')),
        ('short pattern', lambda: ChordTracrRuleModel(
            ('x',), patterns={'x': ('M', 'm')})),
        ('bad quality symbol', lambda: ChordTracrRuleModel(
            ('x',), patterns={'x': ('M', 'm', 'X', 'm')})),
    ):
        try:
            fn()
            check(f'{name} is rejected', False, 'no error raised')
        except ValueError:
            check(f'{name} is rejected', True)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
