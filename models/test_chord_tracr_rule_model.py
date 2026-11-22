"""Tests for the v6 chord-cadence rule model.

Run: python -m models.test_chord_tracr_rule_model
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chord_tracr_rule_model import (
    ChordTracrRuleModel, CHORD_NAMES, MINOR_PATTERNS, N_CHORDS, N_ROOTS, N_POS,
    chord_index, chord_root, chord_is_minor,
)
from rasp_program.sequence_rule import OFFSETS

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok  " if cond else "FAIL"} {name}' + (f'   {detail}' if detail else ''))


def main():
    print('vocabulary')
    check('24 chords', N_CHORDS == 24)
    check('major index == root', all(chord_index(r, False) == r for r in range(12)))
    check('minor index == root + 12',
          all(chord_index(r, True) == r + 12 for r in range(12)))
    check('root/minor round-trip',
          all(chord_root(chord_index(r, m)) == r
              and chord_is_minor(chord_index(r, m)) == m
              for r in range(12) for m in (False, True)))
    check('names line up', CHORD_NAMES[6] == 'F#' and CHORD_NAMES[18] == 'F#m')

    print('\nlayout')
    for mode, want in (('current', 28), ('period4', 52), ('seed_broadcast', 52)):
        m = ChordTracrRuleModel(rule_mode=mode)
        check(f'{mode} -> d_model {want}', m.d_model == want, f'{m.d_model}')
    m = ChordTracrRuleModel()
    check('zero parameters', len(list(m.parameters())) == 0)
    check('nothing trainable in the state dict',
          all(not v.requires_grad for v in m.state_dict().values()))
    check('period4 has next, no seed',
          m.sl_next is not None and m.sl_seed is None)
    sb = ChordTracrRuleModel(rule_mode='seed_broadcast')
    check('seed_broadcast has seed, no next',
          sb.sl_seed is not None and sb.sl_next is None)

    print('\nthe cadence table, against a hand-written reference')
    # major tonic -> I IV V I all major; minor tonic -> i iv V i, the V major
    ok = True
    for c in range(N_CHORDS):
        got = [int(x) for x in m.cadence[c]]
        root = chord_root(c)
        if chord_is_minor(c):
            exp = [chord_index((root + o) % 12, b != 2)
                   for b, o in enumerate(OFFSETS)]
        else:
            exp = [chord_index((root + o) % 12, False) for o in OFFSETS]
        if got != exp:
            ok = False
            print(f'       {CHORD_NAMES[c]}: got {got} want {exp}')
    check('all 24 cadences', ok)
    check('F# -> F# B C# F#',
          [CHORD_NAMES[int(x)] for x in m.cadence[chord_index(6, False)]]
          == ['F#', 'B', 'C#', 'F#'])
    check('F#m -> F#m Bm C# F#m',
          [CHORD_NAMES[int(x)] for x in m.cadence[chord_index(6, True)]]
          == ['F#m', 'Bm', 'C#', 'F#m'])
    check('Am -> Am Dm E Am',
          [CHORD_NAMES[int(x)] for x in m.cadence[chord_index(9, True)]]
          == ['Am', 'Dm', 'E', 'Am'])

    print('\nv6\'s premise, and the leak property the held-out set rests on')
    for r in range(N_ROOTS):
        maj, mino = m.cadence[chord_index(r, False)], m.cadence[chord_index(r, True)]
        same_roots = [chord_root(int(x)) for x in maj] == \
                     [chord_root(int(x)) for x in mino]
        shared = sum(int(a) == int(b) for a, b in zip(maj, mino))
        if not (same_roots and shared == 1):
            check(f'root {r}', False, f'same_roots={same_roots} shared={shared}')
            break
    else:
        check('every parallel pair: identical ROOTS, exactly 1 shared CHORD',
              True, 'the V, major in both modes')
    nat = ChordTracrRuleModel(minor_pattern='natural')
    check('natural minor shares 0 chords with its parallel major',
          sum(int(a) == int(b) for a, b in
              zip(nat.cadence[chord_index(0, False)],
                  nat.cadence[chord_index(0, True)])) == 0)
    check("natural minor's v is minor",
          CHORD_NAMES[int(nat.cadence[chord_index(9, True)][2])] == 'Em')

    print('\nrule_hidden blocks')
    B, T = 7, 16
    torch.manual_seed(0)
    c0 = torch.randint(0, N_CHORDS, (B,))
    h = m.build_rule_hidden_analytic(c0, T)
    check('shape', tuple(h.shape) == (B, T, m.d_model), str(tuple(h.shape)))
    check('every value is 0 or 1', bool(((h == 0) | (h == 1)).all()))
    for name, sl in (('cur', m.sl_cur), ('next', m.sl_next), ('phase', m.sl_phase)):
        check(f'{name}: one-hot at every position',
              bool((h[:, :, sl].sum(-1) == 1).all()))
    cur = m.expected(c0, T)
    check('cur block decodes to the cadence',
          torch.equal(h[:, :, m.sl_cur].argmax(-1), cur))
    t = torch.arange(T)
    check('phase block is t % 4',
          torch.equal(h[:, :, m.sl_phase].argmax(-1),
                      (t % N_POS)[None].expand(B, -1)))
    check('next[t] == cur[t+1]',
          torch.equal(h[:, :-1, m.sl_next], h[:, 1:, m.sl_cur]))
    nxt = m.expected(c0, T + 1)
    check('next at the last position continues the cycle',
          torch.equal(h[:, -1, m.sl_next].argmax(-1), nxt[:, T]))
    hsb = sb.build_rule_hidden_analytic(c0, T)
    check('seed_broadcast: seed block is the tonic chord, constant along t',
          torch.equal(hsb[:, :, sb.sl_seed].argmax(-1), c0[:, None].expand(-1, T)))

    print('\nforward')
    chords = m.expected(c0, T)
    check('period4 logits predict the NEXT chord',
          torch.equal(m(chords).argmax(-1), nxt[:, 1:T + 1]))
    logits, h2 = m(chords, return_hidden=True)
    check('forward hidden == build_rule_hidden_analytic', torch.equal(h, h2))
    cu = ChordTracrRuleModel(rule_mode='current')
    check("rule_mode='current' logits give the CURRENT chord",
          torch.equal(cu(chords).argmax(-1), cur))
    shifted = chords.clone()
    shifted[:, 0] = (shifted[:, 0] + 7) % N_CHORDS
    check('the tonic chord is read from position 0',
          not torch.equal(m(shifted).argmax(-1), m(chords).argmax(-1)))

    print('\nvalidation')
    for name, fn in (
        ('bad rule_mode', lambda: ChordTracrRuleModel(rule_mode='period8')),
        ('bad minor_pattern', lambda: ChordTracrRuleModel(minor_pattern='dorian')),
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
