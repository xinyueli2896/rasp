#!/usr/bin/env python3
"""Does the PROMPT follow the I-IV-V-I rule?

The model reads the key off the prompt. If the prompt itself is off-rule, the
key it infers is wrong through no fault of the adapter — and every generated
slot inherits that error. The two data paths differ here:

  direct       filter_nottingham repairs every accepted window to 8/8 correct
               (_repair_window is applied to the training tensor, not just to
               the saved MIDI), so the prompt always follows the rule.

  orchestrated extract_orchestrated only checks a GLOBAL fraction:
                   match_frac = (# correct slots) / (# detected slots)
                   reject if match_frac < rule_min_frac        # default 0.75
               Nothing constrains WHERE the wrong slots fall, so at 0.75 up to
               2 of 8 may be wrong and both can sit inside the prompt.

This script measures it on the actual .pt datasets, reporting per-slot
compliance and — the number that matters — how often the prompt is clean.

Usage:
  python -m midi_adapter.check_prompt_rule \\
      --data /l/users/xinyue.li/data/pop909_ivvi_w1/val_all_keys_seenkeys.pt \\
             /l/users/xinyue.li/data/pop909_ivvi_w1/direct_val_seenkeys.pt \\
      --n_prompt_beats 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.filter_nottingham import _extract_chord_roots_from_cp
from midi_adapter.evaluate_on_real import _load_windows
from rasp_program.sequence_rule import OFFSETS

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def analyse(pt_path: str, window_len: int, n_prompt_beats: int,
            chords_per_bar: int) -> dict:
    windows, keys, _ = _load_windows(pt_path, window_len)
    spc            = 16 // chords_per_bar
    n_slots        = window_len // spc
    prompt_slots   = n_prompt_beats // spc

    per_slot_ok = np.zeros(n_slots)
    per_slot_n  = np.zeros(n_slots)
    prompt_clean = []
    n_wrong_hist = np.zeros(n_slots + 1)

    for w, key in zip(windows, keys):
        if key < 0:
            continue
        detected = _extract_chord_roots_from_cp(w.numpy(),
                                                chords_per_bar=chords_per_bar)
        wrong = 0
        ok_prompt = True
        for c, d in enumerate(detected[:n_slots]):
            if d < 0:                       # empty slot — not scoreable
                continue
            exp = (key + OFFSETS[c % 4]) % 12
            hit = (d == exp)
            per_slot_n[c] += 1
            per_slot_ok[c] += hit
            if not hit:
                wrong += 1
                if c < prompt_slots:
                    ok_prompt = False
        prompt_clean.append(ok_prompt)
        n_wrong_hist[min(wrong, n_slots)] += 1

    return {
        'name': os.path.basename(pt_path),
        'n': len(prompt_clean),
        'per_slot': per_slot_ok / np.maximum(per_slot_n, 1),
        'prompt_clean': float(np.mean(prompt_clean)) if prompt_clean else float('nan'),
        'n_wrong_hist': n_wrong_hist,
        'prompt_slots': prompt_slots,
        'n_slots': n_slots,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, nargs='+', required=True)
    p.add_argument('--window_len', type=int, default=64)
    p.add_argument('--n_prompt_beats', type=int, default=16)
    p.add_argument('--chords_per_bar', type=int, default=2)
    args = p.parse_args()

    results = []
    for path in args.data:
        if not os.path.exists(path):
            print(f'!! missing: {path}')
            continue
        results.append(analyse(path, args.window_len, args.n_prompt_beats,
                                args.chords_per_bar))

    if not results:
        return
    n_slots = results[0]['n_slots']
    ps      = results[0]['prompt_slots']

    print(f'\nPer-slot rule compliance   (slots 0..{ps-1} are the PROMPT)')
    hdr = ''.join(f'{c:>7}' for c in range(n_slots))
    print(f'  {"dataset":<32}{"n":>6}{hdr}')
    print('  ' + '-' * (32 + 6 + 7 * n_slots))
    for r in results:
        row = ''.join(f'{v:>7.3f}' for v in r['per_slot'])
        print(f'  {r["name"]:<32}{r["n"]:>6}{row}')
    print(f'  {"":<38}' + ''.join(f'{"^" if c < ps else " ":>7}'
                                   for c in range(n_slots)) + '   prompt')

    print(f'\nPrompt clean  (every prompt slot matches the rule)')
    for r in results:
        print(f'  {r["name"]:<32}{r["prompt_clean"]:>8.3f}')

    print(f'\nWrong slots per window')
    print(f'  {"dataset":<32}' + ''.join(f'{k:>7}' for k in range(n_slots + 1)))
    for r in results:
        h = r['n_wrong_hist'] / max(r['n_wrong_hist'].sum(), 1)
        print(f'  {r["name"]:<32}' + ''.join(f'{v:>7.3f}' for v in h))

    print('\nA prompt that is off-rule gives the model misleading evidence about')
    print('the key, so a low "prompt clean" caps key_found no matter how good')
    print('the adapter is.')


if __name__ == '__main__':
    main()
