"""Tests for the compiled v6 chord rule.

Run: python -m models.test_rasp_compiled_chord_rule
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rasp_compiled_chord_rule import (
    RaspCompiledChordRule, VIEW_WIDTH, KEY_BASE, N_CHORDS, build_cadence,
)
from models.chord_tracr_rule_model import ChordTracrRuleModel


def main():
    ok = True
    tonics = torch.arange(N_CHORDS)
    for pattern in ('harmonic', 'natural'):
        for ks in ('first', 'phase0'):
            for spc in (1, 16):
                m = RaspCompiledChordRule(subbeats_per_chord=spc, shift=1, input_shift=-1,
                                          view='chord_next', key_select=ks,
                                          minor_pattern=pattern)
                ref = ChordTracrRuleModel(rule_mode='period4', minor_pattern=pattern,
                                          subbeats_per_chord=spc, shift=1,
                                          input_shift=-1, view='chord_next')
                T = 8 * spc
                a = m.build_rule_hidden_analytic(tonics, T)
                b = ref.build_rule_hidden_analytic(tonics, T)
                err = float((a - b).abs().max())
                good = err < 1e-4
                ok &= good
                print(f"  [{'ok ' if good else 'FAIL'}] {pattern:<8} {ks:<6} spc={spc:<2} "
                      f'compiled vs lookup max |diff| {err:.2e}')
    # The head reads the OBSERVED tonic: a stream starting on Cm retrieves Cm.
    m = RaspCompiledChordRule(view='key_phase', key_select='first')
    _, h = m(torch.tensor([[12, 5, 7, 12]]), return_hidden=True)
    good = int(h[0, 3, KEY_BASE:KEY_BASE + N_CHORDS].argmax()) == 12
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] head retrieves the observed tonic chord")
    good = VIEW_WIDTH['chord_next'] == 24 and sum(p.numel() for p in m.parameters()) == 0
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] chord_next is 24 wide, zero parameters")
    good = int(build_cadence('harmonic')[12 + 6, 2]) == 1   # F#m bar 2 -> C# major
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] harmonic F#m cadence has a major V (C#)")
    print('ALL OK' if ok else 'FAILURES')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
