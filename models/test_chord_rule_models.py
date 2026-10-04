#!/usr/bin/env python3
"""Standalone tests for the compiled chord rule models.

Exercises the REAL torch classes, not a numpy mirror, so a divergence between
what was designed and what was written shows up here rather than after a
40k-step training run.

Run:  python3 models/test_chord_rule_models.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chord_tracr_rule_model import (
    ChordTracrRuleModel, ChordRaspCompiled, N_ROOTS, N_POS,
)
from rasp_program.sequence_rule import OFFSETS

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SPC, T = 8, 64
PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def roots_for(key, T=T, spc=SPC):
    phase = (torch.arange(T) // spc) % N_POS
    return (key + torch.tensor([OFFSETS[p] for p in phase])) % N_ROOTS, phase


def enc_of(m, idx):
    """The model's own encoding of a root index (one-hot or triad)."""
    return m.W_E[idx][..., :N_ROOTS]


# ───────────────────────────────────────────────────────────────────────
print('\n1. ChordTracrRuleModel — retrieve program, all head counts')
for nh in (1, 2, 3, 4):
    ok, worst = True, 1.0
    for key in range(12):
        r, _ = roots_for(key)
        m = ChordTracrRuleModel(subbeats_per_chord=SPC, n_phase_heads=nh)
        _, h = m(r.unsqueeze(0), return_hidden=True)
        tonic = h[0, :, N_ROOTS:2 * N_ROOTS]
        ok &= bool((tonic.argmax(-1) == key).all())
        s = tonic.sort(-1).values
        worst = min(worst, float((s[:, -1] - s[:, -2]).min()))
    check(f'n_phase_heads={nh}: tonic exact for all 12 keys', ok,
          f'worst margin {worst:.3f}')

print('\n2. BOS default — a head with no matching key must contribute nothing')
m = ChordTracrRuleModel(subbeats_per_chord=SPC, n_phase_heads=4)
r, phase = roots_for(7)
x = m._stream_from_roots(r.unsqueeze(0))
# head 1 selects phase 1, whose first position is subbeat 8
h1 = m._head(x, m.W_Q_1, m.W_V_1)
pre  = float(h1[0, :SPC, N_ROOTS:2 * N_ROOTS].abs().max())    # before any phase-1 key
post = float(h1[0, SPC:2 * SPC, N_ROOTS:2 * N_ROOTS].abs().max())
check('unmatched queries emit ~0', pre < 1e-3, f'max |out| before a match = {pre:.2e}')
check('matched queries emit a value', post > 0.9, f'max |out| after = {post:.3f}')

print('\n3. ChordRaspCompiled — full program, both encodings')
for rin in ('root', 'triad'):
    ok = True
    for key in range(12):
        r, _ = roots_for(key)
        m = ChordRaspCompiled(subbeats_per_chord=SPC, rule_input=rin)
        _, h = m(r.unsqueeze(0), return_hidden=True)
        out = h[0, :, m.O0:m.O0 + N_ROOTS]
        ok &= torch.allclose((out > 0.5).float(), enc_of(m, r), atol=1e-5)
    check(f'rule_input={rin}: out == rule(key, phase), all 12 keys', ok)

m = ChordRaspCompiled(subbeats_per_chord=SPC, rule_input='triad')
r, phase = roots_for(7)
_, h = m(r.unsqueeze(0), return_hidden=True)
chords = [[NAMES[i] for i in range(12) if h[0, t, m.O0 + i] > 0.5] for t in range(0, T, SPC)]
check('triad example (key G)', chords[:4] == [['D','G','B'],['C','E','G'],['D','F#','A'],['D','G','B']],
      f'{chords[:4]}')

print('\n4. mlp_only — no head, the key is supplied directly')
for rin in ('root', 'triad'):
    ok = True
    for key in range(12):
        m = ChordRaspCompiled(subbeats_per_chord=SPC, rule_input=rin, use_attention=False)
        r, phase = roots_for(key)
        x = torch.zeros(1, T, m.d_model)
        x[0, :, m.K0:m.K0 + N_ROOTS] = enc_of(m, torch.tensor(key))   # key region
        x[0, :, m.P0:m.P0 + N_POS] = F.one_hot(phase, N_POS).float()
        out = m.run_attention(x)[0, :, m.O0:m.O0 + N_ROOTS]
        ok &= torch.allclose((out > 0.5).float(), enc_of(m, r), atol=1e-5)
    check(f'rule_input={rin}: out == rule(key, phase), all 12 keys', ok)

print('\n5. Causality — r[:t] must not change when the future changes')
for cls, kw in ((ChordTracrRuleModel, dict(n_phase_heads=2)),
                (ChordRaspCompiled, dict(rule_input='triad'))):
    m = cls(subbeats_per_chord=SPC, **kw)
    r, _ = roots_for(7)
    a = m.run_attention(m._stream_from_roots(r.unsqueeze(0)))
    r2 = r.clone(); r2[40:] = (r2[40:] + 3) % N_ROOTS            # perturb the future
    b = m.run_attention(m._stream_from_roots(r2.unsqueeze(0)))
    check(f'{cls.__name__}: positions < 40 unchanged',
          torch.allclose(a[:, :40], b[:, :40], atol=1e-5),
          f'max delta {float((a[:, :40] - b[:, :40]).abs().max()):.2e}')

print('\n6. The MLP transposes an ARBITRARY pitch-class set')
m = ChordRaspCompiled(subbeats_per_chord=SPC)
phase_full = (torch.arange(T) // SPC) % N_POS
def transpose_check(pcs):
    x = torch.zeros(1, T, m.d_model)
    v = torch.zeros(N_ROOTS); v[list(pcs)] = 1.0
    x[0, :, m.K0:m.K0 + N_ROOTS] = v
    x[0, :, m.P0:m.P0 + N_POS] = F.one_hot(phase_full, N_POS).float()
    out = m.run_attention(x)[0, :, m.O0:m.O0 + N_ROOTS]
    return all(torch.allclose(out[t], torch.roll(v, OFFSETS[int(phase_full[t])]),
                              atol=1e-5) for t in range(T))
CHORDS = {'major': {7,11,2}, 'minor': {7,10,2}, 'dom7': {7,11,2,5},
          'maj7': {7,11,2,6}, 'sus4': {7,0,2}, 'single': {7},
          'six-note': {7,11,2,5,9,0}}
check('transposes every chord type exactly',
      all(transpose_check(p) for p in CHORDS.values()),
      ', '.join(CHORDS))

print('\n6b. Linear in the key for values in [0,1]; broken outside')
def err(kv, ph=1):
    x = torch.zeros(1, T, m.d_model)
    x[0, :, m.K0:m.K0 + N_ROOTS] = kv
    x[0, :, m.P0 + ph] = 1.0
    out = m.run_attention(x)[0, 0, m.O0:m.O0 + N_ROOTS]
    return float((out - torch.roll(kv, OFFSETS[ph])).abs().max())
torch.manual_seed(0)
check('softmax key is transposed exactly', err(F.softmax(torch.randn(12), -1)) < 1e-6)
check('arbitrary [0,1] key is transposed exactly', err(torch.rand(12)) < 1e-6)
check('raw logits DO break it', err(torch.randn(12) * 2) > 0.1,
      'which is why proxy_activation=none is rejected for MLP programs')

print('\n7. build_rule_hidden_analytic matches the program it describes')
# The analytic builder returns an idealised one-hot. The program cannot: the
# BOS key sits at 0.5*attn_scale against a match at 1.0*attn_scale, so it keeps
# exp(-0.5*attn_scale) of the softmax mass -- exp(-10) = 4.54e-05 at
# attn_scale=20. That is the price of the default-output mechanism, and it
# shrinks to exp(-50) ~ 2e-22 at Tracr's own coldness of 100.
BOS_LEAK = float(torch.exp(torch.tensor(-0.5 * 20.0)))
print(f'       expected BOS leak at attn_scale=20: {BOS_LEAK:.2e}')
for cls, kw in ((ChordTracrRuleModel, dict(n_phase_heads=1)),
                (ChordRaspCompiled, dict(rule_input='root')),
                (ChordRaspCompiled, dict(rule_input='triad'))):
    m = cls(subbeats_per_chord=SPC, **kw)
    ok, worst = True, 0.0
    for key in range(12):
        r, _ = roots_for(key)
        prog = m.run_attention(m._stream_from_roots(r.unsqueeze(0)))
        ana  = m.build_rule_hidden_analytic(torch.tensor([key]), T, 'cpu')
        d = float((prog - ana).abs().max()); worst = max(worst, d)
        ok &= d <= 2 * BOS_LEAK
    check(f'{cls.__name__}({kw})', ok,
          f'max delta {worst:.2e} (<= 2x BOS leak)')

m = ChordRaspCompiled(subbeats_per_chord=SPC)
for scale in (20.0, 50.0, 100.0):
    m.attn_scale = scale
    r, _ = roots_for(7)
    prog = m.run_attention(m._stream_from_roots(r.unsqueeze(0)))
    ana  = m.build_rule_hidden_analytic(torch.tensor([7]), T, 'cpu')
    print(f'       attn_scale={scale:>5}  ->  max delta '
          f'{float((prog - ana).abs().max()):.2e}')

print('\n8. Zero trainable parameters')
for cls, kw in ((ChordTracrRuleModel, {}), (ChordRaspCompiled, {})):
    m = cls(subbeats_per_chord=SPC, **kw)
    check(f'{cls.__name__} has no parameters', len(list(m.parameters())) == 0)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1)
