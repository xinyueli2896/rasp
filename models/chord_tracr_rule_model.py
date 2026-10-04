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

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importable as a module and runnable directly (python models/<this>.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        n_phase_heads:      int   = 1,
    ):
        super().__init__()
        assert 1 <= n_phase_heads <= N_POS, \
            f'n_phase_heads must be 1..{N_POS}, got {n_phase_heads}'
        self.subbeats_per_chord = subbeats_per_chord
        self.attn_scale         = attn_scale
        self.n_phase_heads      = n_phase_heads
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

        # ── extra phase heads ──────────────────────────────────────────
        # Head 0 (W_Q/W_V above) reads the key off the I chords, where
        # OFFSETS[0] = 0 so the root IS the key. Every other slot also carries
        # the key, just rotated: key = (root - OFFSETS[p]) mod 12. A rotation
        # is a permutation matrix, so head p can un-rotate its own slots for
        # free and recover the same key. Summing the heads turns a
        # single-source lookup into a 4-way ensemble, which matters because a
        # 1-bar prompt contains exactly ONE phase-0 slot — misread it and the
        # 1-head retrieval is wrong for the whole window.
        for p in range(1, n_phase_heads):
            Wq = torch.zeros(d, d)
            for i in range(N_POS):
                Wq[pos_base + p, pos_base + i] = 1.0     # select phase == p
            self.register_buffer(f'W_Q_{p}', Wq)
            Wv = torch.zeros(d, d)
            for i in range(V):
                Wv[V + ((i - OFFSETS[p]) % V), i] = 1.0  # un-rotate to the key
            self.register_buffer(f'W_V_{p}', Wv)

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

    def _head(self, x: torch.Tensor, W_Q: torch.Tensor,
              W_V: torch.Tensor) -> torch.Tensor:
        """One compiled head: select by phase, aggregate the un-rotated root.

    Tracr's BOS convention is implemented here as a virtual key whose logit is
    0.5 * attn_scale and whose value is zero. A query with a MATCHING real key
    scores 1.0 * attn_scale and dominates it; a query with no match scores 0
    everywhere and falls to the virtual key, so the head contributes its
    default (nothing) instead of averaging the whole causal prefix. Without it,
    head p injects garbage at every position before the first phase-p slot --
    with n_phase_heads=2 that is a dead tie between the true key and
    (key - 5) mod 12 over the entire first slot.
        """
        T = x.shape[1]
        Q = x @ W_Q.T
        K = x @ self.W_K.T
        V = x @ W_V.T
        scores = (Q @ K.transpose(-2, -1)) * self.attn_scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        scores = torch.cat(
            [scores.new_full(scores.shape[:-1] + (1,), 0.5 * self.attn_scale),
             scores], dim=-1)
        attn = F.softmax(scores, dim=-1)[..., 1:]     # BOS carries a zero value
        return (attn @ V) @ self.W_O.T

    def run_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run the compiled head(s) over an ARBITRARY residual stream.

        x : (B, T, 28) — a learned proxy, or an analytically built stream.
        Returns (B, T, 28) with dims 12-23 carrying the retrieved tonic.

        With n_phase_heads = 1 this is the seed_broadcast program: read the
        key off the I chords. With 4, each head un-rotates its own phase, so
        the tonic is averaged over all four slot types — the same answer when
        the proxy is clean, and far more robust when it is not. Corrupting the
        prompt's single phase-0 slot takes the 1-head retrieval from exact to
        zero; the 4-head version is unaffected.

        Causal, so position t never reads rule state derived from the future.
        """
        out = self._head(x, self.W_Q, self.W_V)
        for p in range(1, self.n_phase_heads):
            out = out + self._head(x, getattr(self, f'W_Q_{p}'),
                                   getattr(self, f'W_V_{p}'))
        return x + out / self.n_phase_heads

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


RASP_D_MODEL = N_ROOTS * 2 + N_POS + N_ROOTS      # 40


class ChordRaspCompiled(nn.Module):
    """The I-IV-V-I rule compiled the ordinary TracR way: no hand-placed
    shortcuts, every matrix a plain read of one named variable.

    RASP program
    ------------
        lookup = Select(phase, phase, lambda k, q: k == 0)
        key    = Aggregate(lookup, root)                       -> attention
        out    = SequenceMap(lambda k, p: (k + OFFSETS[p]) % 12,
                             key, phase)                       -> MLP

    `Aggregate` compiles to an attention head; `SequenceMap` is an elementwise
    function of two variables at the same position and compiles to an MLP —
    never to attention. ChordTracrRuleModel implements only the first line and
    leaves the arithmetic to the adapter; this class compiles BOTH, so the
    program emits the correct root itself.

    Residual stream, d_model = 40, one disjoint subspace per RASP variable:
        0-11   root            the observed/estimated chord root
        12-23  key             written by the attention sublayer
        24-27  phase           the bar-phase clock
        28-39  out             written by the MLP sublayer = the rule's answer

    MLP construction is the standard categorical SequenceMap:
        h[i,j] = ReLU(key_i + phase_j - 1)      one unit per (key, phase) pair,
                                                 1 exactly when both match
        out    = sum_ij h[i,j] * onehot((i + OFFSETS[j]) % 12)

    Zero trainable parameters.
    """

    d_model: int = RASP_D_MODEL
    CHORD_INTERVALS: tuple = (0, 4, 7)      # major triad

    R0, K0, P0, O0 = 0, N_ROOTS, N_ROOTS * 2, N_ROOTS * 2 + N_POS

    def __init__(self, subbeats_per_chord: int = 8, attn_scale: float = 20.0,
                 rule_input: str = 'root', use_attention: bool = True):
        super().__init__()
        # use_attention=False is the MLP-only program: the proxy writes the KEY
        # region directly, so there is nothing to retrieve and only the
        # SequenceMap remains. Aggregate is essential when the key must be
        # found among supplied chords, but in the no-input variant it forces
        # the key estimate to come from phase-0 positions -- the earliest and
        # least-informed in the window, starting with one that has seen no
        # music at all -- while discarding the context-rich later ones.
        self.use_attention = use_attention
        assert rule_input in ('root', 'triad'), \
            f"rule_input must be 'root' or 'triad', got {rule_input!r}"
        self.subbeats_per_chord = subbeats_per_chord
        self.attn_scale         = attn_scale
        self.rule_input         = rule_input
        V, P, d = N_ROOTS, N_POS, RASP_D_MODEL
        R0, K0, P0, O0 = self.R0, self.K0, self.P0, self.O0

        # 'root'  : W_E[r] is the one-hot of r — one active dim.
        # 'triad' : W_E[r] is the major-triad chromagram {r, r+4, r+7} — three.
        #
        # Both work end to end, because W_out maps each active key dim i to
        # (i + OFFSETS[p]) % 12 — a per-dimension transposition. The rule is
        # itself a transposition, so a 3-hot tonic triad comes out as the
        # 3-hot triad of the correct root. The program transposes a whole
        # pitch-class SET, not just a root.
        #
        # The cost is that exactness needs a CLEAN categorical input. The MLP
        # fires one hidden cell per active key dim, so a 4-hot chromagram
        # produces four transposed notes and the answer is wrong. Measured:
        # 3-hot exact, 4-hot and beyond not. Pair this with
        # --proxy_activation hard if the chromagram comes from a learned
        # projection rather than a lookup.
        W_E = torch.zeros(V, d)
        for r in range(V):
            if rule_input == 'root':
                W_E[r, R0 + r] = 1.0
            else:
                for k in self.CHORD_INTERVALS:
                    W_E[r, R0 + (r + k) % V] = 1.0
        self.register_buffer('W_E', W_E)
        W_pos = torch.zeros(P, d)
        for j in range(P):
            W_pos[j, P0 + j] = 1.0
        self.register_buffer('W_pos', W_pos)

        # ── attention sublayer: Select(phase, phase, k == 0) ──────────────
        W_Q = torch.zeros(d, d)
        for j in range(P):
            W_Q[P0 + 0, P0 + j] = 1.0       # query-side variable: phase
        W_K = torch.zeros(d, d)
        for j in range(P):
            W_K[P0 + j, P0 + j] = 1.0       # key-side variable: phase
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[K0 + i, R0 + i] = 1.0       # value variable: root -> key slot
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[K0 + i, K0 + i] = 1.0
        for n, m in (('W_Q', W_Q), ('W_K', W_K), ('W_V', W_V), ('W_O', W_O)):
            self.register_buffer(n, m)

        # ── MLP sublayer: SequenceMap over (key, phase) ───────────────────
        H = V * P
        W_in = torch.zeros(H, d)
        b_in = torch.full((H,), -1.0)
        for i in range(V):
            for j in range(P):
                W_in[i * P + j, K0 + i] = 1.0
                W_in[i * P + j, P0 + j] = 1.0
        W_out = torch.zeros(d, H)
        for i in range(V):
            for j in range(P):
                W_out[O0 + (i + OFFSETS[j]) % V, i * P + j] = 1.0
        for n, m in (('W_in', W_in), ('b_in', b_in), ('W_out', W_out)):
            self.register_buffer(n, m)

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))

    # ------------------------------------------------------------------

    def phase_at(self, T: int, device) -> torch.Tensor:
        t = torch.arange(T, device=device)
        return (t // self.subbeats_per_chord) % N_POS

    def pos_embed(self, T: int, device) -> torch.Tensor:
        return self.W_pos[self.phase_at(T, device)].unsqueeze(0)

    def run_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run the whole compiled program: attention sublayer, then MLP.

        x : (B, T, 40) with the root and phase subspaces populated and the
            key / out subspaces empty.
        Returns (B, T, 40); dims 28-39 carry the rule's answer.
        """
        if self.use_attention:
            T = x.shape[1]
            Q, K, V = x @ self.W_Q.T, x @ self.W_K.T, x @ self.W_V.T
            scores  = (Q @ K.transpose(-2, -1)) * self.attn_scale
            causal  = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores  = scores.masked_fill(~causal, float('-inf'))
            # Tracr BOS: virtual key at 0.5*coldness with a zero value, so an
            # unmatched query emits the default instead of averaging the prefix.
            scores  = torch.cat(
                [scores.new_full(scores.shape[:-1] + (1,), 0.5 * self.attn_scale),
                 scores], dim=-1)
            attn = F.softmax(scores, dim=-1)[..., 1:]
            x = x + (attn @ V) @ self.W_O.T                       # -> key
        # else: the key region was supplied directly; go straight to the MLP.
        h = F.relu(F.linear(x, self.W_in, self.b_in))             # (B, T, 48)
        return x + h @ self.W_out.T                                # -> out

    def _stream_from_roots(self, roots: torch.Tensor) -> torch.Tensor:
        return self.W_E[roots] + self.pos_embed(roots.shape[1], roots.device)

    def forward(self, roots: torch.Tensor, return_hidden: bool = False):
        h_out  = self.run_attention(self._stream_from_roots(roots))
        logits = h_out[:, :, self.O0:self.O0 + N_ROOTS]
        return (logits, h_out) if return_hidden else logits

    def build_rule_hidden_analytic(self, key: torch.Tensor, T: int,
                                   device) -> torch.Tensor:
        key   = key.to(device)
        phase = self.phase_at(T, device)
        root  = (key[:, None] + self._offsets.to(device)[phase][None, :]) % N_ROOTS
        h = self.W_E[root] + self.pos_embed(T, device)
        # key / out carry the SAME encoding the root region uses: a one-hot in
        # 'root' mode, a triad chromagram in 'triad' mode.
        enc = self.W_E[:, self.R0:self.R0 + N_ROOTS]        # (12, 12)
        h[..., self.K0:self.K0 + N_ROOTS] = enc[key][:, None, :].expand(-1, T, -1)
        h[..., self.O0:self.O0 + N_ROOTS] = enc[root]
        return h

    def build_rule_hidden_from_chord_seq(self, chord_seq: torch.Tensor):
        saved, self.subbeats_per_chord = self.subbeats_per_chord, 1
        try:
            return self.run_attention(self._stream_from_roots(chord_seq))
        finally:
            self.subbeats_per_chord = saved

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    """Compile the flagship rule model and run it. No training, no data."""
    torch.set_printoptions(linewidth=200, sci_mode=False)
    NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    SPC, T, KEY = 8, 64, 7                      # 8 subbeats/slot, 4 bars, key G

    m = ChordRaspCompiled(subbeats_per_chord=SPC)   # <- compiles here
    R0, K0, P0, O0 = m.R0, m.K0, m.P0, m.O0

    print(f'compiled  d_model={m.d_model}  params={len(list(m.parameters()))}')
    print(f'  layout   root {R0}-{K0-1} | key {K0}-{P0-1} | '
          f'phase {P0}-{O0-1} | out {O0}-{m.d_model-1}')
    print(f'  rule     OFFSETS = {m._offsets.tolist()}  (lives in W_out only)')
    print()
    print('nonzeros per matrix')
    for n in ('W_E', 'W_pos', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_in', 'W_out'):
        W = getattr(m, n)
        print(f'  {n:<6}{str(tuple(W.shape)):<12}{int((W != 0).sum()):>5} nonzero')

    # ── run it ──────────────────────────────────────────────────────────
    phase = m.phase_at(T, 'cpu')
    roots = (KEY + m._offsets[phase]) % N_ROOTS            # ground-truth input
    x     = m._stream_from_roots(roots.unsqueeze(0))
    out   = m.run_attention(x)

    print()
    print(f'input: the chord root at every subbeat, key {NAMES[KEY]}')
    print(f'  slot   {[t // SPC for t in range(0, T, SPC)]}')
    print(f'  phase  {phase[::SPC].tolist()}')
    print(f'  root   {[NAMES[r] for r in roots[::SPC]]}')
    print()
    print('output')
    print(f'  key    {[NAMES[int(out[0, t, K0:P0].argmax())] for t in range(0, T, SPC)]}'
          f"   <- retrieved by the attention head")
    print(f'  out    {[NAMES[int(out[0, t, O0:].argmax())] for t in range(0, T, SPC)]}'
          f"   <- computed by the MLP")
    print(f'  truth  {[NAMES[r] for r in roots[::SPC]]}')

    # ── and it is key-agnostic: same weights, any key ────────────────────
    print()
    print('same compiled weights, every key:')
    for k in (0, 3, 7, 11):
        r = (k + m._offsets[phase]) % N_ROOTS
        o = m.run_attention(m._stream_from_roots(r.unsqueeze(0)))
        got = [NAMES[int(o[0, t, O0:].argmax())] for t in range(0, T, SPC)][:4]
        print(f'  key {NAMES[k]:<3} -> {got}')

    print()
    print('swap the cadence by rewriting W_out alone:')
    for offs, name in (([0, 5, 7, 0], 'I-IV-V-I'), ([0, 9, 5, 7], 'I-vi-IV-V')):
        W = torch.zeros_like(m.W_out)
        for i in range(N_ROOTS):
            for j in range(N_POS):
                W[O0 + (i + offs[j]) % N_ROOTS, i * N_POS + j] = 1.0
        m.W_out.copy_(W); m._offsets.copy_(torch.tensor(offs))
        r = (KEY + m._offsets[phase]) % N_ROOTS
        o = m.run_attention(m._stream_from_roots(r.unsqueeze(0)))
        print(f'  {str(offs):<14}{name:<12}'
              f'{[NAMES[int(o[0, t, O0:].argmax())] for t in range(0, T, SPC)][:4]}')
