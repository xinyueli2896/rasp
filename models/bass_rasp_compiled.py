"""
Complete TracR compilation of the bass rule, in PyTorch.

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
D_MODEL   = 44

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


class BassRaspCompiled(nn.Module):
    """One attention head + one MLP: the whole bass rule, compiled."""

    D_MODEL: int = D_MODEL

    def __init__(
        self,
        subbeats_per_chord: int   = 1,
        shift:              int   = 1,
        attn_scale:         float = 100.0,
        bos_beta:           float = 0.5,
        view:               str   = 'key_phase',
    ):
        """
        subbeats_per_chord : positions per phase slot.  1 for the bass
            experiment (one root per beat); 8 would be chord granularity.
        shift : how far ahead the SequenceMap looks.  1 emits the NEXT root
            (what an AR adapter at position t needs), 0 the current root.
        attn_scale : Tracr's inverse temperature, 100 in the paper.
        bos_beta : beta_BOS from Appendix D.1.  0.5 = attend BOS only when
            nothing else is selected.
        view : which subspaces rule_hidden() returns.
        """
        super().__init__()
        if view not in VIEWS:
            raise ValueError(f'view must be one of {sorted(VIEWS)}, got {view!r}')
        self.subbeats_per_chord = subbeats_per_chord
        self.shift      = shift
        self.attn_scale = attn_scale
        self.bos_beta   = bos_beta
        self.view       = view
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

        # --- attention: Select(phase, phase, k == 0), Aggregate(tokens) ---
        W_Q = torch.zeros(d, d)
        W_K = torch.zeros(d, d)
        for p in range(N_POS):
            W_Q[PH_BASE + 0, PH_BASE + p] = 1.0   # every query -> phase-0 dir
            W_K[PH_BASE + p, PH_BASE + p] = 1.0   # identity on phase
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

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        T, dev = idx.shape[1], idx.device
        return (self.W_E[idx]
                + self.W_phase[self.phase_at(T, dev, 0)].unsqueeze(0)
                + self.W_phase_next[self.phase_at(T, dev, self.shift)].unsqueeze(0))

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

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    print(f'd_model = {D_MODEL}   views: '
          + ', '.join(f'{k}={v}' for k, v in VIEW_WIDTH.items()))
    for view in VIEWS:
        m = BassRaspCompiled(view=view)
        idx = torch.tensor([[(6 + OFFSETS[t % 4]) % 12 for t in range(8)]])
        rh = m.rule_hidden(idx)
        print(f'  {view:<10} rule_d_model={m.rule_d_model:<3} '
              f'rule_hidden={tuple(rh.shape)}')
