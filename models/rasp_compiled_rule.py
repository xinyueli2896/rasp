"""
Complete TracR compilation of the I-IV-V-I rule, in PyTorch.

Shared by BOTH experiments. The integer task
    token[i] = (x + OFFSETS[i % 4]) % 12
and the CP bass task
    root[t]  = (key + OFFSETS[t % 4]) % 12
are the same RASP program over the same vocabulary (12) with the same
offsets; only the carrier differs (an integer token vs a bass note).
One compiled model therefore serves both, with subbeats_per_chord=1.

BassTracrRuleModel compiles only half the RASP program: an attention head that
retrieves the key.  The arithmetic (key, phase) -> root is done in Python
integer ops and then one-hot embedded:

    cur_r   = (key[:, None] + o_cur[None, :]) % N_ROOTS
    tok_cur = self.W_E[cur_r]

so the rule never enters the weights, and there is no MLP in any of its modes.
This module compiles the whole program, following Tracr (Lindner et al., 2023,
arXiv:2301.05062) and github.com/google-deepmind/tracr.

The program
-----------
    phase      = indices // spc % 4                       positional
    phase_next = (indices + shift) // spc % 4             positional
    key        = Aggregate(Select(phase, phase, k == 0), tokens)
    root_next  = SequenceMap(lambda k, p: (k + OFFSETS[p]) % 12, key, phase_next)

Two phases, not one, because the Select and the SequenceMap need different
ones: Select must match phase-0 slots at their true positions, while the
SequenceMap must produce the root for position t+shift.  Both are functions of
`indices` alone, so neither leaks: shifting by +1 lets the model emit the NEXT
root causally, which is what an autoregressive adapter at position t needs.
Set shift=0 to emit the current root instead.

Operator -> component (paper S3 step 3: "elementwise operations become MLP
blocks, and select-aggregate operations become attention blocks"):

    phase, phase_next   Map(indices)   -> MLP, but constant-folded into W_pos
                                          (both depend only on position, so
                                          the MLP that would compute them is a
                                          constant).  This is a deliberate
                                          deviation from what Tracr emits: it
                                          saves one block, and the compiled
                                          model is one layer rather than two.
    Select + Aggregate                 -> attention head (W_QK, W_OV)
    SequenceMap                        -> MLP (W_in/b_in, W_out)

Residual stream, one orthogonal subspace per s-op (paper S3 step 5), d = 44:

    dims  0-11 : tokens      one-hot of the OBSERVED pitch class at t
    dims 12-23 : key         written by the attention head
    dims 24-27 : phase       positional, drives the Select
    dims 28-31 : phase_next  positional, drives the SequenceMap
    dims 32-43 : root_next   written by the MLP -- the program's output

The MLP (paper Appendix D.1)
----------------------------
"SequenceMaps with categorical inputs and outputs become MLPs where the first
layer maps to an encoding of all pairs of inputs and the second layer acts as
a lookup table."  48 pair neurons, each reading one key dim and one phase_next
dim with bias -1 so ReLU fires only when both are on; the second layer sends
each pair to one_hot((k + OFFSETS[p]) % 12).

BOS (paper Appendix D.1)
------------------------
Tracr adds a rank-one update W_BOS = beta * x_one * x_bos^T with beta = 1/2 so
BOS is attended whenever no other token is selected.  Implemented as a virtual
key of logit 0.5 * attn_scale carrying a zero value, so a query that selects
nothing contributes nothing instead of averaging its causal prefix.

Known deviation
---------------
Appendix D.1 says categorical values are only valid "if the head never attends
to more than one token".  Select(phase == 0) matches EVERY phase-0 slot, so at
t=8 it attends t in {0, 4, 8}.  On rule-following input those carry identical
tokens, so the average is still a clean one-hot; on corrupted input it is a
blend.  The alternative, Select(indices, indices, k == 0), attends exactly one
token and satisfies the constraint, but needs `indices` one-hot in the stream,
making d_model grow with context length (424 at TRAIN_LENGTH=384 versus 44
here) and making retrieval depend on a single note that can never be
outvoted.

No layer norm (paper S3).  No trainable parameters: every weight is a buffer.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasp_program.sequence_rule import OFFSETS

N_ROOTS = 12
N_POS   = 4

TOK_BASE  = 0                  # dims  0-11  tokens
KEY_BASE  = 12                 # dims 12-23  key
PH_BASE   = 24                 # dims 24-27  phase
PHN_BASE  = 28                 # dims 28-31  phase_next
ROOT_BASE = 32                 # dims 32-43  root_next
FIRST_BASE = 44                # dim  44     marks the slot holding beat 0
D_MODEL   = 45                 # no VIEW exposes dim 44, so the view widths
                               # (12/16/32/44) are unchanged by it

# Which subspaces are handed to the cross-attention adapter as rule_hidden.
#   root_next : the answer -- the adapter only has to copy it
#   key_phase : key and phase_next -- the adapter must do the arithmetic
#   computed  : everything the network computed, minus the raw token echo
#   full      : the entire residual stream
VIEWS: dict[str, list[tuple[int, int]]] = {
    'root_next': [(ROOT_BASE, N_ROOTS)],
    'key_phase': [(KEY_BASE, N_ROOTS), (PHN_BASE, N_POS)],
    'computed':  [(KEY_BASE, N_ROOTS), (PH_BASE, N_POS),
                  (PHN_BASE, N_POS), (ROOT_BASE, N_ROOTS)],
    'full':      [(TOK_BASE, N_ROOTS), (KEY_BASE, N_ROOTS), (PH_BASE, N_POS),
                  (PHN_BASE, N_POS), (ROOT_BASE, N_ROOTS)],
}

VIEW_WIDTH = {name: sum(n for _, n in spans) for name, spans in VIEWS.items()}
# root_next 12 | key_phase 16 | computed 32 | full 44


def view_dims(view: str) -> list[int]:
    """The residual-stream dimensions a view selects, in order."""
    if view not in VIEWS:
        raise ValueError(f'view must be one of {sorted(VIEWS)}, got {view!r}')
    return [b + i for b, n in VIEWS[view] for i in range(n)]


class RaspCompiledRule(nn.Module):
    """One attention head + one MLP: the whole bass rule, compiled."""

    D_MODEL: int = D_MODEL

    def __init__(
        self,
        subbeats_per_chord: int   = 1,
        shift:              int   = 1,
        input_shift:        int   = 0,
        attn_scale:         float = 100.0,
        bos_beta:           float = 0.5,
        view:               str   = 'key_phase',
        key_select:         str   = 'phase0',
    ):
        """
        subbeats_per_chord : positions per phase slot.  1 for the bass
            experiment (one root per beat); 8 would be chord granularity.
        shift : how far ahead the SequenceMap looks, relative to input_shift.
        input_shift : what the token at stream index j actually is.  0 means
            index j holds the value at j (the integer task).  -1 means index j
            holds the value at j-1, which is the case when the host shifts its
            stream by one before the blocks, as the CP transformer does with
            h = cat([sos, h[:-1]]).  Select then matches phase-0 slots at
            phase_at(input_shift) and the SequenceMap emits for
            phase_at(input_shift + shift), so both stay positional and the
            output lands on the position the adapter is about to produce.
        attn_scale : Tracr's inverse temperature, 100 in the paper.
        bos_beta : beta_BOS from Appendix D.1.  0.5 = attend BOS only when
            nothing else is selected.
        view : which subspaces rule_hidden() returns.
        key_select : how the head finds the key.
            'phase0' Select(phase, phase, k == 0) -- matches EVERY phase-0
                slot, so at t=9 it averages beats 0, 4 and 8. On
                rule-following input they are identical and the average is a
                clean one-hot, but it violates Appendix D.1's requirement
                that a categorical head attend to at most one token, and off
                the rule manifold it returns a blend.
            'first'  Select(is_first, is_first, k == 1) -- matches exactly the
                one slot holding beat 0, satisfying D.1. Costs one extra
                residual dimension, which is positional and therefore
                constant-folded into the embedding like `phase`. The
                trade-off is the one the paper should state: retrieval now
                rests on a single note that can never be outvoted.
        """
        if key_select not in ('phase0', 'first'):
            raise ValueError(f"key_select must be 'phase0' or 'first', got {key_select!r}")
        super().__init__()
        if view not in VIEWS:
            raise ValueError(f'view must be one of {sorted(VIEWS)}, got {view!r}')
        self.subbeats_per_chord = subbeats_per_chord
        self.shift       = shift
        self.input_shift = input_shift
        self.attn_scale = attn_scale
        self.bos_beta   = bos_beta
        self.view       = view
        self.key_select = key_select
        self.d_model    = D_MODEL
        self.rule_d_model = VIEW_WIDTH[view]

        V, d = N_ROOTS, D_MODEL

        W_E = torch.zeros(V, d)
        W_E[:, TOK_BASE:TOK_BASE + V] = torch.eye(V)
        self.register_buffer('W_E', W_E)

        W_phase = torch.zeros(N_POS, d)
        W_phase_next = torch.zeros(N_POS, d)
        for p in range(N_POS):
            W_phase[p, PH_BASE + p]       = 1.0
            W_phase_next[p, PHN_BASE + p] = 1.0
        self.register_buffer('W_phase', W_phase)
        self.register_buffer('W_phase_next', W_phase_next)

        # is_first: 1 at the stream slot holding beat 0. The beat at stream
        # index t is beat t + input_shift, so beat 0 sits at t = -input_shift
        # (index 0 for the integer host, index 1 for the bos-shifted CP host).
        # Like phase, it is a function of position alone -- the Map that would
        # compute it folds into the embedding.
        self.first_index = -input_shift

        # --- attention: Select(phase, phase, k == 0), Aggregate(tokens) ---
        W_Q = torch.zeros(d, d)
        W_K = torch.zeros(d, d)
        if key_select == 'phase0':
            for p in range(N_POS):
                W_Q[PH_BASE + 0, PH_BASE + p] = 1.0   # every query -> phase-0 dir
                W_K[PH_BASE + p, PH_BASE + p] = 1.0   # identity on phase
        else:
            # Every query points along is_first; only the beat-0 slot carries
            # it, so the score is 1 for exactly one key and 0 for all others.
            for p in range(N_POS):
                W_Q[FIRST_BASE, PH_BASE + p] = 1.0
            W_K[FIRST_BASE, FIRST_BASE] = 1.0
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)

        W_V = torch.zeros(d, d)
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_V[KEY_BASE + i, TOK_BASE + i] = 1.0
            W_O[KEY_BASE + i, KEY_BASE + i] = 1.0
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)

        # --- MLP: SequenceMap((key, phase_next) -> root_next) -------------
        n_pairs = V * N_POS
        W_in  = torch.zeros(n_pairs, d)
        b_in  = torch.full((n_pairs,), -1.0)
        W_out = torch.zeros(d, n_pairs)
        for k in range(V):
            for p in range(N_POS):
                n = k * N_POS + p
                W_in[n, KEY_BASE + k]  = 1.0
                W_in[n, PHN_BASE + p]  = 1.0
                W_out[ROOT_BASE + (k + OFFSETS[p]) % V, n] = 1.0
        self.register_buffer('W_in',  W_in)
        self.register_buffer('b_in',  b_in)
        self.register_buffer('W_out', W_out)

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))
        self.register_buffer('_view_dims', torch.tensor(view_dims(view),
                                                        dtype=torch.long))

    # ------------------------------------------------------------------

    def phase_at(self, T: int, device, shift: int = 0) -> torch.Tensor:
        t = torch.arange(T, device=device) + shift
        return (t // self.subbeats_per_chord) % N_POS

    def first_marker(self, T: int, device) -> torch.Tensor:
        """(T, d_model): a 1 in the is_first dim at the slot holding beat 0.

        Positional, so it is part of the embedding rather than something the
        network computes. Zero everywhere under key_select='phase0', where
        nothing reads it.
        """
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
        """The compiled head, over an arbitrary float residual stream."""
        T = x.shape[1]
        Q, K, V_ = x @ self.W_Q.t(), x @ self.W_K.t(), x @ self.W_V.t()
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        # -1e30 rather than -inf: an all-masked row would give NaN, and the
        # BOS column is what is meant to carry those rows.
        scores = scores.masked_fill(~causal, -1e30)
        bos = scores.new_full(scores.shape[:-1] + (1,),
                              self.bos_beta * self.attn_scale)
        attn = F.softmax(torch.cat([bos, scores], dim=-1), dim=-1)[..., 1:]
        return x + (attn @ V_) @ self.W_O.t()

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        """The compiled SequenceMap: pair encoding, then lookup table."""
        return x + F.relu(x @ self.W_in.t() + self.b_in) @ self.W_out.t()

    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """idx: (B, T) observed pitch classes -> root_next logits (B, T, 12)."""
        h = self.mlp(self.attention(self.embed(idx)))
        logits = h[:, :, ROOT_BASE:ROOT_BASE + N_ROOTS]
        if return_hidden:
            return logits, h
        return logits

    def rule_hidden(self, idx: torch.Tensor,
                    view: str | None = None) -> torch.Tensor:
        """(B, T, rule_d_model) -- the subspaces handed to the adapter."""
        _, h = self(idx, return_hidden=True)
        dims = (self._view_dims if view is None
                else torch.tensor(view_dims(view), device=h.device))
        return h.index_select(-1, dims)

    # ------------------------------------------------------------------

    def build_rule_hidden_analytic(self, pc_0: torch.Tensor, T: int,
                                   device, view: str | None = None) -> torch.Tensor:
        """rule_hidden for a KNOWN key, by RUNNING the compiled program.

        The sampling path knows the key from the prompt but needs the whole
        window up front, before the notes it describes exist. Only the input
        is reconstructed here: under the rule the slot at stream index t
        carries (key + OFFSETS[phase(t)]) % 12, so the stream the compiled
        model would have read can be written down from pc_0 alone. Everything
        after that is the real forward pass -- embed, the attention head that
        retrieves the key, and the MLP that computes the root.

        This used to write the key and root one-hots directly, which made the
        generation path a closed-form stand-in rather than the compiled model:
        neither W_Q/W_K/W_V/W_O nor W_in/W_out executed at inference, and the
        head's retrieval behaviour (key_select) could not matter. The output
        is unchanged -- it agreed with the forward pass by construction, and
        the tests assert it still does -- but the claim that a compiled
        transformer supplies the signal is now literally true at inference,
        and key_select is exercised where it is actually used.

        Index 0 is zeroed when input_shift < 0: the host has not seen a beat
        there, so nothing is derivable. The token placed at that slot is never
        read -- its phase label is (0 - 1) % 4 = 3, which no Select matches,
        and it carries no is_first marker.
        """
        pc_0 = pc_0.to(device).long()
        ph   = self.phase_at(T, device, self.input_shift)

        # The stream the rule implies, which is what the model would have read.
        tok_r = (pc_0[:, None] + self._offsets.to(device)[ph][None, :]) % N_ROOTS

        h = self.mlp(self.attention(self.embed(tok_r)))
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


if __name__ == '__main__':
    print(f'd_model = {D_MODEL}   views: '
          + ', '.join(f'{k}={v}' for k, v in VIEW_WIDTH.items()))
    for view in VIEWS:
        m = RaspCompiledRule(view=view)
        idx = torch.tensor([[(6 + OFFSETS[t % 4]) % 12 for t in range(8)]])
        rh = m.rule_hidden(idx)
        print(f'  {view:<10} rule_d_model={m.rule_d_model:<3} '
              f'rule_hidden={tuple(rh.shape)}')


# Backwards-compatible alias: this started life as a bass-only module.
BassRaspCompiled = RaspCompiledRule
