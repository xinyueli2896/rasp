"""
The v6 chord cadence as a compiled RASP program, built like v5's rule model.

v5 (models/rasp_compiled_rule.py) compiles

    key       = Aggregate(Select(is_first, is_first, k == 1), tokens)
    root_next = SequenceMap(lambda k, p: (k + OFFSETS[p]) % 12, key, phase_next)

into one attention head and one MLP over a 12-symbol vocabulary of pitch
classes. This module compiles the SAME program over the 24 triads:

    tonic      = Aggregate(Select(is_first, is_first, k == 1), tokens)
    chord_next = SequenceMap(lambda c, p: CADENCE[c][p], tonic, phase_next)

    CADENCE[F#]  = F#  B   C#  F#        I  - IV - V - I
    CADENCE[F#m] = F#m Bm  C#  F#m       i  - iv - V - i   (harmonic: V major)

Nothing else changes. The token at each stream position is the OBSERVED
chord there -- the bass pitch class at that subbeat, exactly v5's token, plus
the quality of the bar it sits in -- the attention head retrieves the tonic
chord from it, and the MLP computes the next bar's chord from (tonic,
phase_next). As in v5, generation reconstructs the stream the rule implies
from the prompt's tonic chord and runs the real forward pass over it, so the
compiled weights execute at inference too.

Operator -> component, as in v5 (Lindner et al. 2023, S3):

    phase, phase_next, is_first   Map(indices)  -> constant-folded into the
                                                   positional embedding
    Select + Aggregate            -> attention head (W_Q, W_K, W_V, W_O)
    SequenceMap                   -> MLP: 24 x 4 = 96 pair neurons, each
                                     reading one tonic dim and one phase_next
                                     dim with bias -1, then a lookup to
                                     one_hot(CADENCE[c][p])  (paper App. D.1)

Residual stream, one orthogonal subspace per s-op, d = 81:

    dims  0-23 : tokens      one-hot observed chord at t
    dims 24-47 : tonic       written by the attention head
    dims 48-51 : phase       positional, drives the Select under 'phase0'
    dims 52-55 : phase_next  positional, drives the SequenceMap
    dims 56-79 : chord_next  written by the MLP -- the program's output
    dim  80    : is_first    marks the slot holding beat 0

Views handed to the adapter mirror v5's:
    chord_next  24   the answer (v5's root_next)
    key_phase   28   tonic + phase_next: the adapter must do the lookup
    computed    56   everything the network computed
    full        80   the whole stream minus is_first

BOS, attention scale and the 'phase0' / 'first' key selection are v5's,
including its documented deviation for 'phase0'. No trainable parameters:
every weight is a non-persistent buffer, so checkpoints carry none of it.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasp_program.sequence_rule import OFFSETS
from models.chord_tracr_rule_model import (
    MINOR_PATTERNS, N_CHORDS, N_ROOTS, chord_index,
)

N_POS = len(OFFSETS)

TOK_BASE   = 0
KEY_BASE   = TOK_BASE + N_CHORDS          # 24
PH_BASE    = KEY_BASE + N_CHORDS          # 48
PHN_BASE   = PH_BASE + N_POS              # 52
CHORD_BASE = PHN_BASE + N_POS             # 56
FIRST_BASE = CHORD_BASE + N_CHORDS        # 80
D_MODEL    = FIRST_BASE + 1               # 81

VIEWS: dict[str, list[tuple[int, int]]] = {
    'chord_next': [(CHORD_BASE, N_CHORDS)],
    'key_phase':  [(KEY_BASE, N_CHORDS), (PHN_BASE, N_POS)],
    'computed':   [(KEY_BASE, N_CHORDS), (PH_BASE, N_POS),
                   (PHN_BASE, N_POS), (CHORD_BASE, N_CHORDS)],
    'full':       [(TOK_BASE, N_CHORDS), (KEY_BASE, N_CHORDS), (PH_BASE, N_POS),
                   (PHN_BASE, N_POS), (CHORD_BASE, N_CHORDS)],
}
VIEW_WIDTH = {name: sum(n for _, n in spans) for name, spans in VIEWS.items()}
# chord_next 24 | key_phase 28 | computed 56 | full 80


def view_dims(view: str) -> list[int]:
    if view not in VIEWS:
        raise ValueError(f'view must be one of {sorted(VIEWS)}, got {view!r}')
    return [b + i for b, n in VIEWS[view] for i in range(n)]


def build_cadence(minor_pattern: str = 'harmonic') -> torch.Tensor:
    """(24, 4) CADENCE[tonic chord, bar] -> chord index. Same table as v6's."""
    quals = MINOR_PATTERNS[minor_pattern]
    cad = torch.zeros(N_CHORDS, N_POS, dtype=torch.long)
    for c in range(N_CHORDS):
        root, minor = c % N_ROOTS, c >= N_ROOTS
        for b in range(N_POS):
            cad[c, b] = chord_index((root + OFFSETS[b]) % N_ROOTS,
                                    bool(minor and quals[b] == 1))
    return cad


class RaspCompiledChordRule(nn.Module):
    """One attention head + one MLP: the v6 chord cadence, compiled."""

    D_MODEL: int = D_MODEL

    def __init__(
        self,
        subbeats_per_chord: int   = 1,
        shift:              int   = 1,
        input_shift:        int   = 0,
        attn_scale:         float = 100.0,
        bos_beta:           float = 0.5,
        view:               str   = 'chord_next',
        key_select:         str   = 'first',
        minor_pattern:      str   = 'harmonic',
    ):
        """Arguments are v5's RaspCompiledRule arguments, with view names
        over chords (chord_next replaces root_next) and the minor pattern
        that fixes the V's quality."""
        if key_select not in ('phase0', 'first'):
            raise ValueError(f"key_select must be 'phase0' or 'first', got {key_select!r}")
        if view not in VIEWS:
            raise ValueError(f'view must be one of {sorted(VIEWS)}, got {view!r}')
        if minor_pattern not in MINOR_PATTERNS:
            raise ValueError(f'minor_pattern must be one of {sorted(MINOR_PATTERNS)}')
        super().__init__()
        self.subbeats_per_chord = int(subbeats_per_chord)
        self.shift = int(shift)
        self.input_shift = int(input_shift)
        self.attn_scale = attn_scale
        self.bos_beta = bos_beta
        self.view = view
        self.key_select = key_select
        self.minor_pattern = minor_pattern
        self.d_model = D_MODEL
        self.rule_d_model = VIEW_WIDTH[view]
        self.first_index = -self.input_shift

        V, d = N_CHORDS, D_MODEL
        cadence = build_cadence(minor_pattern)
        self.register_buffer('cadence', cadence, persistent=False)

        W_E = torch.zeros(V, d)
        W_E[:, TOK_BASE:TOK_BASE + V] = torch.eye(V)
        self.register_buffer('W_E', W_E, persistent=False)

        W_phase = torch.zeros(N_POS, d)
        W_phase_next = torch.zeros(N_POS, d)
        for p in range(N_POS):
            W_phase[p, PH_BASE + p] = 1.0
            W_phase_next[p, PHN_BASE + p] = 1.0
        self.register_buffer('W_phase', W_phase, persistent=False)
        self.register_buffer('W_phase_next', W_phase_next, persistent=False)

        # --- attention: Select(...), Aggregate(tokens) -> tonic ------------
        W_Q = torch.zeros(d, d)
        W_K = torch.zeros(d, d)
        if key_select == 'phase0':
            for p in range(N_POS):
                W_Q[PH_BASE + 0, PH_BASE + p] = 1.0
                W_K[PH_BASE + p, PH_BASE + p] = 1.0
        else:
            for p in range(N_POS):
                W_Q[FIRST_BASE, PH_BASE + p] = 1.0
            W_K[FIRST_BASE, FIRST_BASE] = 1.0
        self.register_buffer('W_Q', W_Q, persistent=False)
        self.register_buffer('W_K', W_K, persistent=False)

        W_V = torch.zeros(d, d)
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_V[KEY_BASE + i, TOK_BASE + i] = 1.0
            W_O[KEY_BASE + i, KEY_BASE + i] = 1.0
        self.register_buffer('W_V', W_V, persistent=False)
        self.register_buffer('W_O', W_O, persistent=False)

        # --- MLP: SequenceMap((tonic, phase_next) -> chord_next) -----------
        n_pairs = V * N_POS
        W_in = torch.zeros(n_pairs, d)
        b_in = torch.full((n_pairs,), -1.0)
        W_out = torch.zeros(d, n_pairs)
        for c in range(V):
            for p in range(N_POS):
                n = c * N_POS + p
                W_in[n, KEY_BASE + c] = 1.0
                W_in[n, PHN_BASE + p] = 1.0
                W_out[CHORD_BASE + int(cadence[c, p]), n] = 1.0
        self.register_buffer('W_in', W_in, persistent=False)
        self.register_buffer('b_in', b_in, persistent=False)
        self.register_buffer('W_out', W_out, persistent=False)

        self.register_buffer('_view_dims', torch.tensor(view_dims(view)),
                             persistent=False)

    # ------------------------------------------------------------------

    def phase_at(self, T: int, device, shift: int = 0) -> torch.Tensor:
        t = torch.arange(T, device=device) + shift
        return (t // self.subbeats_per_chord) % N_POS

    def first_marker(self, T: int, device) -> torch.Tensor:
        m = torch.zeros(T, self.d_model, device=device)
        if self.key_select == 'first' and 0 <= self.first_index < T:
            m[self.first_index, FIRST_BASE] = 1.0
        return m

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        T, dev = idx.shape[1], idx.device
        return (self.W_E[idx]
                + self.W_phase[self.phase_at(T, dev, self.input_shift)].unsqueeze(0)
                + self.W_phase_next[
                      self.phase_at(T, dev, self.input_shift + self.shift)].unsqueeze(0)
                + self.first_marker(T, dev).unsqueeze(0))

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        Q, K, V_ = x @ self.W_Q.t(), x @ self.W_K.t(), x @ self.W_V.t()
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, -1e30)
        bos = scores.new_full(scores.shape[:-1] + (1,), self.bos_beta * self.attn_scale)
        attn = F.softmax(torch.cat([bos, scores], dim=-1), dim=-1)[..., 1:]
        return x + (attn @ V_) @ self.W_O.t()

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(x @ self.W_in.t() + self.b_in) @ self.W_out.t()

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """idx: (B, T) observed chords 0..23 -> chord_next logits (B, T, 24)."""
        h = self.mlp(self.attention(self.embed(idx.long())))
        logits = h[:, :, CHORD_BASE:CHORD_BASE + N_CHORDS]
        return (logits, h) if return_hidden else logits

    def rule_hidden(self, idx: torch.Tensor, view: str | None = None) -> torch.Tensor:
        """(B, T, rule_d_model): the subspaces handed to the adapter."""
        _, h = self(idx, return_hidden=True)
        dims = (self._view_dims if view is None
                else torch.tensor(view_dims(view), device=h.device))
        return h.index_select(-1, dims)

    def implied_stream(self, chord_0: torch.Tensor, T: int, device) -> torch.Tensor:
        """(B, T) the chord stream the rule implies for tonic chord_0, at the
        stream's own alignment (index t holds beat t + input_shift)."""
        ph = self.phase_at(T, device, self.input_shift)
        return self.cadence.to(device)[chord_0.to(device).long()][:, ph]

    def build_rule_hidden_analytic(self, chord_0: torch.Tensor, T: int,
                                   device=None, view: str | None = None) -> torch.Tensor:
        """rule_hidden for a KNOWN tonic chord, by RUNNING the compiled program.

        v5's method verbatim: reconstruct the input stream the rule implies,
        then embed -> attention head -> MLP. Index 0 is zeroed when
        input_shift < 0, since the host has seen no beat there.
        """
        device = device or chord_0.device
        h = self.mlp(self.attention(self.embed(self.implied_stream(chord_0, T, device))))
        if self.input_shift < 0:
            h = h.clone()
            h[:, 0] = 0.0
        dims = (self._view_dims if view is None
                else torch.tensor(view_dims(view), device=device))
        return h.index_select(-1, dims)

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])
