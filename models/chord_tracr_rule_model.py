"""
TracR-style compiled rule model for the I-IV-V-I chord rule.

Structural analogue of TracrPyTorchRuleModel (integer experiment, mod 12 over
raw positions) lifted to CHORD-SLOT granularity, so the two experiments run on
the same mechanism rather than merely the same story.

Rule:  root[c] = (key + OFFSETS[c % 4]) % 12,   OFFSETS = [0, 5, 7, 0]
       c = chord slot index; at chords_per_bar=2 and beat_div=4 one slot spans
       8 subbeats, so phase(t) = (t // subbeats_per_chord) % 4.

Residual stream, d_model = 12*2 + 4 = 28 — the same layout TracR uses:
    dims  0-11 : one-hot of the CURRENT chord root at this position
    dims 12-23 : one-hot of the retrieved TONIC (the key), written by attention
    dims 24-27 : one-hot of the bar phase (c % 4)

The single frozen attention head implements the RASP program

    lookup = Select(phase, phase, lambda k, q: k == 0)
    key    = Aggregate(lookup, root)

i.e. "attend to every phase-0 slot and copy its root into the tonic subspace".
Phase-0 slots carry the I chord by definition, so the aggregate is the key, and
under a causal mask slot 0 is always reachable. That leaves

    root = (key + OFFSETS[phase]) % 12

for the ADAPTER to compute — exactly the division of labour in the integer
experiment's 'seed_broadcast' mode. The rule model retrieves; it does not solve.

Why this matters for the bidirectional ("no rule input") variant:
CPChordRuleModel is a pure lookup, so a learned ar_to_rule projection into it
is unconstrained and needs an auxiliary loss to stay in rule coordinates. Here
the proxy is pushed through W_Q/W_K/W_V/W_O instead. Those matrices read ONLY
the phase subspace for routing and move ONLY the root subspace as payload, so a
proxy that fails to populate those subspaces cleanly produces a blurred
retrieval (attn_scale=20 makes the head winner-take-all when the phase dims are
sharp, and mush when they are not) and a useless tonic — which the LM loss
penalises. The pin is gradient pressure through frozen structure, the same
mechanism the integer experiment relied on, not a hard projection.

No trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import OFFSETS

N_ROOTS = 12
N_POS   = 4
CHORD_TRACR_D_MODEL = N_ROOTS * 2 + N_POS   # 28


class ChordTracrRuleModel(nn.Module):

    TRACR_D_MODEL:   int   = CHORD_TRACR_D_MODEL
    d_model:         int   = CHORD_TRACR_D_MODEL
    CHORD_INTERVALS: tuple = (0, 4, 7)      # major triad, for chromagram views

    def __init__(
        self,
        subbeats_per_chord: int   = 8,
        attn_scale:         float = 20.0,
    ):
        super().__init__()
        self.subbeats_per_chord = subbeats_per_chord
        self.attn_scale         = attn_scale
        V, d = N_ROOTS, CHORD_TRACR_D_MODEL
        pos_base = d - N_POS                 # 24

        # W_E[r] = one-hot(r) in dims 0..11
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer('W_E', W_E)

        # W_pos[p] = one-hot(p) in dims 24..27
        W_pos = torch.zeros(N_POS, d)
        for i in range(N_POS):
            W_pos[i, pos_base + i] = 1.0
        self.register_buffer('W_pos', W_pos)

        # W_Q: every phase maps to phase-0 in the query → "select k with phase 0"
        W_Q = torch.zeros(d, d)
        for i in range(N_POS):
            W_Q[pos_base + 0, pos_base + i] = 1.0
        self.register_buffer('W_Q', W_Q)

        # W_K: identity on the phase subspace → K[k] encodes phase(k)
        W_K = torch.zeros(d, d)
        for i in range(N_POS):
            W_K[pos_base + i, pos_base + i] = 1.0
        self.register_buffer('W_K', W_K)

        # W_V: payload is the root, moved into the tonic slot
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[V + i, i] = 1.0
        self.register_buffer('W_V', W_V)

        # W_O: identity on the tonic slot
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[V + i, V + i] = 1.0
        self.register_buffer('W_O', W_O)

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def phase_at(self, T: int, device) -> torch.Tensor:
        """(T,) bar phase for each position, at this model's slot width."""
        t = torch.arange(T, device=device)
        return (t // self.subbeats_per_chord) % N_POS

    def pos_embed(self, T: int, device) -> torch.Tensor:
        """(1, T, 28) frozen phase encoding, ready to add to a proxy.

        Phase is the clock, not the rule: it follows from position alone and
        carries no information about the key. Injecting it lets ar_to_rule
        spend its capacity on pitch content, which is the part under test.
        """
        return self.W_pos[self.phase_at(T, device)].unsqueeze(0)

    # ------------------------------------------------------------------
    # The frozen head — the entry point the bidirectional proxy uses
    # ------------------------------------------------------------------

    def run_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run the compiled head over an ARBITRARY residual stream.

        x : (B, T, 28) — a learned proxy, or an analytically built stream.
        Returns (B, T, 28) with dims 12-23 carrying the retrieved tonic.

        Causal, so position t never reads rule state derived from the future.
        """
        T = x.shape[1]
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        scores = (Q @ K.transpose(-2, -1)) * self.attn_scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        return x + (attn @ V) @ self.W_O.T

    # ------------------------------------------------------------------
    # Analytic construction (targets, eval, explicit-input mode)
    # ------------------------------------------------------------------

    def _stream_from_roots(self, roots: torch.Tensor) -> torch.Tensor:
        """(B, T) roots → (B, T, 28) residual stream with phase added."""
        T = roots.shape[1]
        return self.W_E[roots] + self.pos_embed(T, roots.device)

    def forward(self, roots: torch.Tensor, return_hidden: bool = False):
        """roots : (B, T) chord root per position. Runs the real head."""
        h_out  = self.run_attention(self._stream_from_roots(roots))
        logits = h_out[:, :, N_ROOTS:N_ROOTS * 2]     # retrieved tonic
        return (logits, h_out) if return_hidden else logits

    def build_rule_hidden_analytic(self, key: torch.Tensor, T: int,
                                   device) -> torch.Tensor:
        """(B,) key → (B, T, 28), the stream the head would produce.

        dims 0-11 current root, 12-23 the key, 24-27 phase. Used as the
        supervision target and for eval-time reference.
        """
        key   = key.to(device)
        phase = self.phase_at(T, device)                        # (T,)
        root  = (key[:, None] + self._offsets.to(device)[phase][None, :]) % N_ROOTS
        h = self.W_E[root] + self.pos_embed(T, device)          # (B, T, 28)
        # dims 12-23: the tonic the head retrieves
        tonic = F.one_hot(key, num_classes=N_ROOTS).to(h.dtype)  # (B, 12)
        h[..., N_ROOTS:N_ROOTS * 2] = tonic[:, None, :].expand(-1, T, -1)
        return h

    def build_rule_hidden_from_chord_seq(self, chord_seq: torch.Tensor) -> torch.Tensor:
        """(B, N_chords) explicit roots → (B, N_chords, 28) at SLOT granularity.

        Slot-granular input means one position per chord, so the slot width is
        1 here regardless of subbeats_per_chord.
        """
        saved, self.subbeats_per_chord = self.subbeats_per_chord, 1
        try:
            return self.run_attention(self._stream_from_roots(chord_seq))
        finally:
            self.subbeats_per_chord = saved

    def chromagram_of(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, 28) → (B, T, 12) major-triad chromagram of dims 0-11.

        The root one-hot and the triad are related by a fixed circulant, so
        nothing is lost by carrying the root; this is only a convenience view.
        """
        root = h[..., :N_ROOTS]
        return sum(root.roll(i, dims=-1) for i in self.CHORD_INTERVALS).clamp(0, 1)

    # No trainable parameters — mirrors TracrPyTorchRuleModel.
    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)
    spc = 8
    m = ChordTracrRuleModel(subbeats_per_chord=spc)

    # 4-bar window in G (key=7): I IV V I I IV V I over 8 slots / 64 subbeats.
    key = torch.tensor([7])
    T = 64
    phase = m.phase_at(T, 'cpu')
    roots = (key[:, None] + m._offsets[phase][None, :]) % 12

    print('phase  :', phase[::8].tolist(), '(one per slot)')
    print('roots  :', roots[0, ::8].tolist(), '(expect 7 0 2 7 7 0 2 7)')

    logits, h = m(roots, return_hidden=True)
    retrieved = logits.argmax(-1)[0]
    print('tonic  :', retrieved[::8].tolist(), '(expect all 7)')
    print('exact  :', bool((retrieved == 7).all()))

    # The head must be sharp: check the attention actually concentrates on
    # phase-0 positions rather than smearing.
    x = m._stream_from_roots(roots)
    Q, K = x @ m.W_Q.T, x @ m.W_K.T
    s = (Q @ K.transpose(-2, -1)) * m.attn_scale
    s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float('-inf'))
    a = F.softmax(s, dim=-1)[0]
    p0 = (phase == 0)
    print('mass on phase-0 keys, last row: '
          f'{a[-1][p0].sum():.4f}  (expect ~1.0)')

    # Analytic target must match what the head produces.
    tgt = m.build_rule_hidden_analytic(key, T, 'cpu')
    print('analytic == compiled:', bool(torch.allclose(tgt, h, atol=1e-5)))
