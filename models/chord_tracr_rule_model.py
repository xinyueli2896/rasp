"""
Analytical rule model for v6: a CHORD cadence, not a root cadence.

v1-v5 gave the adapter one symbol per step from a 12-symbol vocabulary of
pitch classes, and the rule was a position-indexed lookup over it:

    token[i] = (x + OFFSETS[i % 4]) % 12          OFFSETS = [0, 5, 7, 0]

v6 keeps that shape exactly and widens the vocabulary to the 24 triads --
twelve major, twelve minor. The rule is then a cadence over CHORDS:

    chord[b] = CADENCE[tonic_chord][b % 4]

    F#  ->  F#  B   C#  F#        (I  - IV - V - I)
    F#m ->  F#m Bm  C#  F#m       (i  - iv - V - i)

Those offsets are root intervals and are identical for the major and minor
tonic, so a root-only rule cannot distinguish the two cadences -- which is
what v4 and v5 could not express. Representing the chord as one symbol rather
than a (root, quality) pair is the point: the rule is about harmony, so the
rule's alphabet is harmonies, and "generalise to an unseen key" becomes
"generalise to an unseen tonic chord" with no separate root channel to reason
about.

The practical consequence for a held-out set: under chord symbols a key's
parallel partner shares only ONE of its four chords (the V, which is major in
both modes), where under root symbols it shared all four roots. So holding out
F# major while training F# minor leaks a quarter of the cadence, not the whole
of it -- and the held-out tonic chord still appears in trained cadences at
non-tonic positions, exactly as v1-v5's held-out starters appeared as
non-initial tokens.

Zero parameters, like BassTracrRuleModel, which this leaves untouched.

Vocabulary
----------
    index  0..11   major triads, index == root pitch class
    index 12..23   minor triads, index == root + 12
so chord_root(i) = i % 12 and a minor chord is i >= 12.

Residual stream
---------------
The three layouts mirror BassTracrRuleModel's, with V = 24 instead of 12:

  rule_mode='current'         [ current(24) | phase(4) ]                 d=28
  rule_mode='period4'         [ current(24) | next(24) | phase(4) ]      d=52
  rule_mode='seed_broadcast'  [ current(24) | tonic(24) | phase(4) ]     d=52

The adapter's k_proj/v_proj take rule_dim as their input width, so a v6
adapter cannot load a v4/v5 checkpoint. v6 is a new experiment, not a
continuation; the frozen-host comparison is unaffected.

On being "Tracr-equivalent"
--------------------------
CADENCE is a fixed lookup over (tonic_chord, phase) -- 24 x 4 cells -- which
is the shape a Tracr `SequenceMap` over a broadcast value and a positional one
compiles to: one MLP hidden unit per cell, writing that cell's one-hot chord
into the residual stream. Nothing is learned and nothing is approximated, so
the lookup is written directly rather than simulated with an attention stack.
Broadcasting the tonic chord from the prompt to every position IS an attention
operation, and it is the seed_broadcast pattern BassTracrRuleModel already
implements; here it arrives as an argument, read off the prompt by the caller
(the root from the downbeat bass, the quality from the bar's third).

A factored variant of this model -- separate one-hot root and quality blocks,
d=34 -- is in git history at commit a5dccc4. Keeping both would be worth it
only as an ablation on whether factoring the rule's alphabet helps
compositional generalisation; it is not carried here as dead code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import OFFSETS

N_ROOTS = 12
N_POS = len(OFFSETS)                 # 4 bars to a cycle
N_CHORDS = 2 * N_ROOTS               # 24 triads
ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_NAMES = ([n for n in ROOT_NAMES] + [n + 'm' for n in ROOT_NAMES])

# Which quality each bar of the cycle takes, per tonic quality. The census
# over the LA corpus found harmonic minor (a major V) outnumbering natural
# minor 3381 cycles to 274, so harmonic is the default; 'natural' is kept
# because the rule should be stateable, not assumed.
MINOR_PATTERNS = {'harmonic': (1, 1, 0, 1),     # i  iv  V  i   -- V is MAJOR
                  'natural':  (1, 1, 1, 1)}     # i  iv  v  i
MAJOR_PATTERN = (0, 0, 0, 0)                    # I  IV  V  I
_IS_MINOR = 1


def chord_index(root: int, minor: bool) -> int:
    return root % N_ROOTS + (N_ROOTS if minor else 0)


def chord_root(idx: int) -> int:
    return idx % N_ROOTS


def chord_is_minor(idx: int) -> bool:
    return idx >= N_ROOTS


class ChordTracrRuleModel(nn.Module):
    """chord[b] = CADENCE[tonic_chord][b % 4], over a 24-triad vocabulary."""

    def __init__(self, rule_mode: str = 'period4',
                 minor_pattern: str = 'harmonic'):
        super().__init__()
        if rule_mode not in ('current', 'period4', 'seed_broadcast'):
            raise ValueError("rule_mode must be 'current', 'period4' or "
                             f"'seed_broadcast', got {rule_mode!r}")
        if minor_pattern not in MINOR_PATTERNS:
            raise ValueError(f'minor_pattern must be one of '
                             f'{sorted(MINOR_PATTERNS)}, got {minor_pattern!r}')
        self.rule_mode = rule_mode
        self.minor_pattern = minor_pattern

        off = 0
        self.sl_cur = slice(off, off + N_CHORDS); off += N_CHORDS
        if rule_mode == 'period4':
            self.sl_next = slice(off, off + N_CHORDS); off += N_CHORDS
            self.sl_seed = None
        elif rule_mode == 'seed_broadcast':
            self.sl_seed = slice(off, off + N_CHORDS); off += N_CHORDS
            self.sl_next = None
        else:
            self.sl_next = self.sl_seed = None
        self.sl_phase = slice(off, off + N_POS); off += N_POS
        self.d_model = off

        # CADENCE[tonic_chord, phase] -> chord index. The whole rule, 24 x 4.
        quals = torch.tensor(MINOR_PATTERNS[minor_pattern], dtype=torch.long)
        cadence = torch.zeros(N_CHORDS, N_POS, dtype=torch.long)
        offs = torch.tensor(OFFSETS, dtype=torch.long)
        for c in range(N_CHORDS):
            root = c % N_ROOTS
            pat = quals if c >= N_ROOTS else torch.zeros(N_POS, dtype=torch.long)
            for b in range(N_POS):
                cadence[c, b] = chord_index(int((root + offs[b]) % N_ROOTS),
                                            bool(pat[b] == _IS_MINOR))
        self.register_buffer('cadence', cadence)

    # ------------------------------------------------------------------

    def expected(self, chord_0: torch.Tensor, T: int) -> torch.Tensor:
        """The rule's answer at every position: (B, T) chord indices.

        chord_0 : (B,) the tonic chord, 0..23, read from the prompt.
        The data prep and the metrics score against this, so the rule lives in
        one place.
        """
        t = torch.arange(T, device=chord_0.device) % N_POS
        return self.cadence.to(chord_0.device)[chord_0.long()][:, t]

    def build_rule_hidden_analytic(self, chord_0: torch.Tensor, T: int,
                                   device: torch.device | None = None):
        """(B, T, d_model), mirroring BassTracrRuleModel's method."""
        device = device or chord_0.device
        chord_0 = chord_0.to(device).long()
        B = chord_0.shape[0]
        t = torch.arange(T, device=device)

        cur = self.expected(chord_0, T)
        h = torch.zeros(B, T, self.d_model, device=device)
        h[:, :, self.sl_cur] = F.one_hot(cur, N_CHORDS).float()
        if self.sl_next is not None:
            nxt = self.cadence.to(device)[chord_0][:, (t + 1) % N_POS]
            h[:, :, self.sl_next] = F.one_hot(nxt, N_CHORDS).float()
        if self.sl_seed is not None:
            h[:, :, self.sl_seed] = F.one_hot(chord_0, N_CHORDS).float()[
                :, None, :].expand(-1, T, -1)
        h[:, :, self.sl_phase] = F.one_hot(t % N_POS, N_POS).float()[
            None].expand(B, -1, -1)
        return h

    def forward(self, chords: torch.Tensor, return_hidden: bool = False):
        """chords: (B, T) chord indices; chords[:, 0] is the tonic chord.

        Returns (B, T, 24) logits: the NEXT chord under 'period4' (as
        BassTracrRuleModel's period4 logits are the next pitch class), the
        current one otherwise.
        """
        T = chords.shape[1]
        h = self.build_rule_hidden_analytic(chords[:, 0], T, chords.device)
        logits = h[:, :, self.sl_next if self.sl_next is not None else self.sl_cur]
        return (logits, h) if return_hidden else logits

    # ------------------------------------------------------------------

    def describe(self) -> str:
        lines = [f'ChordTracrRuleModel(rule_mode={self.rule_mode!r}, '
                 f'minor_pattern={self.minor_pattern!r}, '
                 f'd_model={self.d_model}, vocab={N_CHORDS})',
                 f'  OFFSETS {OFFSETS}']
        for c in range(N_CHORDS):
            row = ' '.join(CHORD_NAMES[int(x)] for x in self.cadence[c])
            lines.append(f'  {CHORD_NAMES[c]:<4} -> {row}')
        return '\n'.join(lines)

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    m = ChordTracrRuleModel()
    print(m.describe())
    c0 = torch.tensor([chord_index(6, False), chord_index(6, True)])   # F#, F#m
    seq = m.expected(c0, 8)
    for i in range(2):
        print(f'  {CHORD_NAMES[int(c0[i])]:<4} '
              + ' '.join(CHORD_NAMES[int(x)] for x in seq[i]))
    h = m.build_rule_hidden_analytic(c0, 8)
    print(f'  rule_hidden {tuple(h.shape)}, parameters '
          f'{sum(1 for _ in m.parameters())}')
