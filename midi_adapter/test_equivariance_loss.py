#!/usr/bin/env python3
"""Checks on CPYinyangTransformer.equivariance_loss.

Three things can silently be wrong, and all three would leave training looking
healthy while teaching the encoder nothing:

  1. the ROTATION DIRECTION. target[r] = base[(r - s) % 12] is the correct
     convention; the sign flip is also a perfectly smooth loss that converges
     to an ANTI-equivariant read.
  2. SHIFT LEGALITY. If s is ever drawn into a held-out key the unseen-key
     result is contaminated and the whole experiment is void.
  3. the MIDI-RANGE GUARD. preprocess() folds the shift into a token index, so
     a note pushed past 127 bleeds into the next duration bucket instead of
     erroring.

Rather than stand up the 111M base, these bind the real method onto a stub that
reimplements preprocess()'s pitch field exactly (`pitch + (dur + 1) * 128 +
shift * is_not_drum`) and swaps _chroma_logits_at for a read whose equivariance
is known by construction.

  python -m midi_adapter.test_equivariance_loss
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_yinyang import CPYinyangTransformer

TUPLES, T, B = 4, 16, 8
N_ROOTS = 12


class StubBase:
    """preprocess() only, matching cp_transformer's with_velocity=False branch."""

    def preprocess(self, x, pitch_shift, tuple_size=4):
        Bq, Tq, S = x.shape
        x = x.long().view(Bq, Tq, S // 4, 4)
        out = torch.zeros(Bq, Tq, S // 4, 2, dtype=torch.long)
        pad = x[..., 1] == 255
        is_not_drum = x[..., 0] != 127
        out[..., 0] = x[..., 0]
        out[..., 1] = (x[..., 1] + (x[..., 2] + 1) * 128
                       + pitch_shift[:, None, None].to(x.device) * is_not_drum)
        out[pad] = 0
        return out.view(Bq, Tq, S // 4 * 2)


class Stub:
    """Minimum surface equivariance_loss touches."""

    def __init__(self, chroma, rule_from_layer=0):
        self.base            = StubBase()
        self.bidirectional   = True
        self.rule_from_layer = rule_from_layer
        self._chroma_logits_at = chroma
        self.seen_shifts: list[int] = []
        self.equivariance_loss = types.MethodType(
            CPYinyangTransformer.equivariance_loss, self)


def make_x(lo=50, hi=80, seed=0):
    """Raw CP windows: program 0..126, pitch in [lo, hi], one pad per step."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, T, TUPLES * 4, dtype=torch.long)
    x[..., 0::4] = torch.randint(0, 127, (B, T, TUPLES), generator=g)
    x[..., 1::4] = torch.randint(lo, hi + 1, (B, T, TUPLES), generator=g)
    x[..., 2::4] = torch.randint(0, 8, (B, T, TUPLES), generator=g)
    # a pad tuple in the last slot of every step
    x[..., -4] = 255
    x[..., -3] = 255
    return x


def pc_oracle(x_proc, layer):
    """An EXACTLY equivariant read: the first tuple's pitch class, one-hot.

    Transposing the music by s sends every pitch class p -> (p + s) % 12, so
    this read must rotate by s. preprocess() put the shift inside the pitch
    field, so %128 %12 recovers the transposed class.
    """
    pc = (x_proc[..., 1] % 128) % 12
    return F.one_hot(pc.long(), N_ROOTS).float() * 10.0


def main():
    fails = []

    def check(name, cond, detail=''):
        print(f'  {"PASS" if cond else "FAIL"}  {name}'
              + (f'   {detail}' if detail else ''))
        if not cond:
            fails.append(name)

    x  = make_x()
    ps = torch.zeros(B, dtype=torch.long)
    # every sample in C, so +-1 (C#/B) are both legal with F#/G# held out
    key = torch.zeros(B, dtype=torch.long)

    print('\n1. rotation convention')
    eq = Stub(pc_oracle).equivariance_loss(x, ps, key)
    check('an exactly equivariant read scores ~0',
          eq is not None and float(eq) < 1e-9, f'loss={float(eq):.3e}')

    # Same read, rotated the WRONG way: if the convention in the method were
    # flipped, THIS is what would score 0 instead.
    def anti(x_proc, layer):
        return pc_oracle(x_proc, layer).roll(1, dims=-1)

    eq_anti = Stub(anti).equivariance_loss(x, ps, key)
    check('a read rotated the wrong way still scores ~0 (rotation is relative)',
          float(eq_anti) < 1e-9, f'loss={float(eq_anti):.3e}')

    # The discriminating case: a read that does NOT commute with transposition.
    g = torch.Generator().manual_seed(1)
    W = torch.randn(N_ROOTS, N_ROOTS, generator=g)

    def non_equivariant(x_proc, layer):
        return pc_oracle(x_proc, layer) @ W

    eq_bad = Stub(non_equivariant).equivariance_loss(x, ps, key)
    check('a non-equivariant read is penalised',
          float(eq_bad) > 1.0, f'loss={float(eq_bad):.3f}')

    # A constant read is trivially equivariant only if it is also constant
    # across the 12 dims; a non-uniform constant is NOT, and must be caught --
    # this is the collapse mode worth knowing about.
    def const_nonuniform(x_proc, layer):
        v = torch.arange(N_ROOTS).float()
        return v.expand(x_proc.shape[0], x_proc.shape[1], N_ROOTS)

    eq_const = Stub(const_nonuniform).equivariance_loss(x, ps, key)
    check('a non-uniform CONSTANT read is penalised (collapse is not a free win)',
          float(eq_const) > 1.0, f'loss={float(eq_const):.3f}')

    def const_uniform(x_proc, layer):
        return torch.zeros(x_proc.shape[0], x_proc.shape[1], N_ROOTS)

    eq_zero = Stub(const_uniform).equivariance_loss(x, ps, key)
    check('a ZERO read scores 0 — equivariance alone cannot prevent collapse, '
          'the CE term has to', float(eq_zero) < 1e-9)

    print('\n2. shift legality (the leak that would void the experiment)')
    # Record the shift actually handed to preprocess.
    seen: list[torch.Tensor] = []

    def spy_stub(keys, unseen, shifts):
        s = Stub(pc_oracle)
        real = s.base.preprocess

        def wrapped(xx, pshift, tuple_size=4):
            seen.append(pshift.clone())
            return real(xx, pshift, tuple_size)

        s.base.preprocess = wrapped
        s.equivariance_loss(x, ps, keys, unseen_keys=unseen, shifts=shifts)
        # call 0 is the unshifted pass, call 1 carries s
        return seen[-1] - seen[-2]

    # F (=5): +1 lands on F#, which is held out, so only -1 is legal
    s_F = spy_stub(torch.full((B,), 5), (6, 8), (-1, 1))
    check('key F never transposes up into F#', bool((s_F == -1).all()),
          f'shifts={sorted(set(s_F.tolist()))}')

    # G (=7): -1 lands on F#, +1 on G# -- both held out, so nothing is legal
    s_G_stub = Stub(pc_oracle)
    eq_G = s_G_stub.equivariance_loss(x, ps, torch.full((B,), 7),
                                      unseen_keys=(6, 8), shifts=(-1, 1))
    check('key G with both neighbours held out contributes nothing',
          eq_G is None)

    # C (=0): both legal, and over many draws both should appear
    got = set()
    for _ in range(40):
        got |= set(spy_stub(torch.zeros(B, dtype=torch.long), (6, 8), (-1, 1)).tolist())
    check('key C draws from both legal shifts', got == {-1, 1}, f'{sorted(got)}')

    # The generator argument the design rests on: s=+1 is reachable from a seen
    # key, so equivariance under the legal shifts extends to all 12 keys.
    seen_keys = [k for k in range(12) if k not in (6, 8)]
    gen_ok = any((k + 1) % 12 in seen_keys for k in seen_keys)
    check('s=+1 is legal from at least one seen key (generates the group)',
          gen_ok)

    print('\n3. MIDI-range guard')
    # Pitches pinned at 127: +1 would overflow into the next duration bucket.
    x_hi = make_x(lo=127, hi=127)
    eq_hi = Stub(pc_oracle).equivariance_loss(
        x_hi, ps, torch.zeros(B, dtype=torch.long), (6, 8), (1,))
    check('a window at pitch 127 rejects s=+1', eq_hi is None)

    x_lo = make_x(lo=0, hi=0)
    eq_lo = Stub(pc_oracle).equivariance_loss(
        x_lo, ps, torch.zeros(B, dtype=torch.long), (6, 8), (-1,))
    check('a window at pitch 0 rejects s=-1', eq_lo is None)

    # Pad tuples (pitch 255) must not be mistaken for out-of-range notes.
    eq_pad = Stub(pc_oracle).equivariance_loss(x, ps, key, (6, 8), (1,))
    check('pad tuples do not trip the range guard', eq_pad is not None)

    print('\n4. gradient reaches the read')
    lin = torch.nn.Linear(N_ROOTS, N_ROOTS)

    def learnable(x_proc, layer):
        return lin(pc_oracle(x_proc, layer))

    s = Stub(learnable)
    out = s.equivariance_loss(x, ps, key)
    out.backward()
    check('loss is differentiable w.r.t. the read',
          lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0)

    print(f'\n{"all checks passed" if not fails else f"{len(fails)} FAILED: {fails}"}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
