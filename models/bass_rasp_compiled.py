"""
Complete TracR compilation of the bass rule, in PyTorch.

BassTracrRuleModel implements only half the RASP program: an attention head
that retrieves the key.  The arithmetic (key, phase) -> root is done in Python
integer ops and then one-hot embedded, so it lives outside the network.  This
module compiles the WHOLE program into weights, following Tracr (Lindner et
al., 2023, arXiv:2301.05062) and github.com/google-deepmind/tracr.

The program
-----------
    phase = indices // subbeats_per_chord % 4          (positional, frozen)
    key   = Aggregate(Select(phase, phase, k == 0), tokens)
    root  = SequenceMap(lambda k, p: (k + OFFSETS[p]) % 12, key, phase)

Two s-ops -> two blocks, which is exactly one transformer layer:

    Select/Aggregate  ->  attention head     (paper S3, "select-aggregate
                          operations become attention blocks")
    SequenceMap       ->  MLP                (paper S3, "elementwise
                          operations become MLP blocks")

Residual stream (paper S3 step 5: "we embed each s-op in its own orthogonal
subspace, which is reserved for its sole use throughout the entire network"):

    dims  0-11 : tokens   one-hot of the OBSERVED pitch class at t   (W_E)
    dims 12-23 : key      written by the attention head
    dims 24-27 : phase    one-hot of the phase                        (W_pos)
    dims 28-39 : root     written by the MLP -- the program's output
    d_model = 40

Dims 0-27 carry the same meaning as BassTracrRuleModel's seed_broadcast mode,
which is precisely the attention half of this program; dims 28-39 are what
that mode leaves for the adapter to compute.

The MLP (paper Appendix D.1)
----------------------------
"SequenceMaps with categorical inputs and outputs become MLPs where the first
layer maps to an encoding of all pairs of inputs and the second layer acts as
a lookup table."

    layer 1 : one neuron per (key, phase) pair -- 12 * 4 = 48 of them.
              W_in reads key dim k and phase dim p with weight 1, bias -1, so
              ReLU(1 + 1 - 1) = 1 iff both are on and ReLU(1 + 0 - 1) = 0
              otherwise.  A one-hot encoding of the pair.
    layer 2 : W_out sends pair neuron (k, p) to one_hot((k + OFFSETS[p]) % 12)
              in the root subspace.  A lookup table.

BOS (paper Appendix D.1)
------------------------
Tracr adds a rank-one update W_BOS = beta * x_one * x_bos^T with beta = 1/2 so
the BOS position is attended whenever no other token is selected.  Implemented
here as a virtual key with logit 0.5 * attn_scale and a zero value vector: a
real match scores 1.0 * attn_scale and dominates it, while a query with no
match falls back to the virtual key and contributes nothing instead of
averaging the whole causal prefix.

Differences from BassTracrRuleModel that matter
-----------------------------------------------
1. W_E embeds the ACTUAL input tokens.  BassTracrRuleModel embeds
   (key + OFFSETS[t % 4]) % 12 recomputed from idx[:, 0], so its token dims
   always show the rule-correct pitch class even when the input does not.
   This model reads what it is given.
2. The arithmetic is in the weights, so the forward pass is differentiable all
   the way through -- a float proxy injected in place of W_E[idx] receives
   gradient through both the head and the MLP.

No layer norm (paper S3: "notably removing layer norms").
No trainable parameters: every weight is a buffer.
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
D_MODEL = N_ROOTS * 3 + N_POS      # 40

TOK_BASE   = 0                      # dims  0-11
KEY_BASE   = N_ROOTS                # dims 12-23
PHASE_BASE = N_ROOTS * 2            # dims 24-27
ROOT_BASE  = N_ROOTS * 2 + N_POS    # dims 28-39


class BassRaspCompiled(nn.Module):
    """One attention head + one MLP: the whole bass rule, compiled."""

    D_MODEL: int = D_MODEL

    def __init__(
        self,
        subbeats_per_chord: int   = 1,
        attn_scale:         float = 100.0,
        bos_beta:           float = 0.5,
    ):
        """
        subbeats_per_chord : positions per phase slot.  1 for the bass
            experiment (one root per beat); 8 would be chord granularity.
        attn_scale : Tracr's inverse temperature, 100 by default in the paper
            ("scale up the matrix by an inverse-temperature parameter T^-1
            (100 by default)").
        bos_beta : beta_BOS from Appendix D.1.  0.5 = attend BOS only when no
            other token is selected.
        """
        super().__init__()
        self.subbeats_per_chord = subbeats_per_chord
        self.attn_scale = attn_scale
        self.bos_beta   = bos_beta
        self.d_model    = D_MODEL

        V, d = N_ROOTS, D_MODEL

        # --- embeddings -------------------------------------------------
        W_E = torch.zeros(V, d)
        W_E[:, TOK_BASE:TOK_BASE + V] = torch.eye(V)
        self.register_buffer('W_E', W_E)

        W_pos = torch.zeros(N_POS, d)
        for p in range(N_POS):
            W_pos[p, PHASE_BASE + p] = 1.0
        self.register_buffer('W_pos', W_pos)

        # --- attention: Select(phase, phase, k == 0) + Aggregate(tokens) --
        # W_QK is low rank, "its row space being the part of the residual
        # stream where the query s-op is stored, and the column space being
        # where the key s-op is stored" (Appendix D.1).  Every query maps to
        # the phase-0 direction, so q.k = 1 exactly when the key has phase 0.
        W_Q = torch.zeros(d, d)
        for p in range(N_POS):
            W_Q[PHASE_BASE + 0, PHASE_BASE + p] = 1.0
        self.register_buffer('W_Q', W_Q)

        W_K = torch.zeros(d, d)
        for p in range(N_POS):
            W_K[PHASE_BASE + p, PHASE_BASE + p] = 1.0
        self.register_buffer('W_K', W_K)

        # W_OV maps the value input (tokens) to the output dims (key).
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[KEY_BASE + i, TOK_BASE + i] = 1.0
        self.register_buffer('W_V', W_V)

        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[KEY_BASE + i, KEY_BASE + i] = 1.0
        self.register_buffer('W_O', W_O)

        # --- MLP: SequenceMap((key, phase) -> root) ----------------------
        n_pairs = V * N_POS                     # 48
        W_in = torch.zeros(n_pairs, d)
        b_in = torch.full((n_pairs,), -1.0)
        W_out = torch.zeros(d, n_pairs)
        for k in range(V):
            for p in range(N_POS):
                n = k * N_POS + p
                W_in[n, KEY_BASE + k]   = 1.0   # fires only when both the
                W_in[n, PHASE_BASE + p] = 1.0   # key and the phase are on
                root = (k + OFFSETS[p]) % V
                W_out[ROOT_BASE + root, n] = 1.0
        self.register_buffer('W_in',  W_in)
        self.register_buffer('b_in',  b_in)
        self.register_buffer('W_out', W_out)

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))

    # ------------------------------------------------------------------

    def phase_at(self, T: int, device) -> torch.Tensor:
        """(T,) phase index for each position."""
        t = torch.arange(T, device=device)
        return (t // self.subbeats_per_chord) % N_POS

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, 40) residual stream, tokens + phase."""
        T = idx.shape[1]
        return self.W_E[idx] + self.W_pos[self.phase_at(T, idx.device)].unsqueeze(0)

    # ------------------------------------------------------------------

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """The compiled head, over an arbitrary residual stream.

        Causal (Appendix D.3).  Takes a float stream, so a learned proxy can
        be fed here in place of embed() and still receive gradient.
        """
        T = x.shape[1]
        Q = x @ self.W_Q.t()
        K = x @ self.W_K.t()
        V_ = x @ self.W_V.t()

        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        # -1e30 rather than -inf: an all-masked row would otherwise produce
        # NaN, and Tracr's BOS column is meant to carry those rows.
        scores = scores.masked_fill(~causal, -1e30)

        # Virtual BOS key: logit beta * attn_scale, value 0.
        bos = scores.new_full(scores.shape[:-1] + (1,), self.bos_beta * self.attn_scale)
        attn = F.softmax(torch.cat([bos, scores], dim=-1), dim=-1)[..., 1:]

        return x + (attn @ V_) @ self.W_O.t()

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        """The compiled SequenceMap: pair encoding, then lookup table."""
        pairs = F.relu(x @ self.W_in.t() + self.b_in)      # (B, T, 48)
        return x + pairs @ self.W_out.t()                  # (B, T, 40)

    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T) observed pitch classes 0-11.

        Returns root logits (B, T, 12) read off the root subspace, and
        optionally the full (B, T, 40) residual stream.
        """
        h = self.embed(idx)
        h = self.attention(h)
        h = self.mlp(h)
        logits = h[:, :, ROOT_BASE:ROOT_BASE + N_ROOTS]
        if return_hidden:
            return logits, h
        return logits

    # ------------------------------------------------------------------

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    m = BassRaspCompiled()
    print(f'd_model = {m.d_model}   trainable params = '
          f'{sum(p.numel() for p in m.parameters())}')
    for key in (0, 6, 8):
        T = 12
        idx = torch.tensor([[(key + OFFSETS[t % 4]) % 12 for t in range(T)]])
        logits, h = m(idx, return_hidden=True)
        print(f'key={key:2d}  in={idx[0].tolist()}')
        print(f'        root={logits[0].argmax(-1).tolist()}  '
              f'key_retrieved={h[0, :, KEY_BASE:KEY_BASE + 12].argmax(-1).tolist()}')
