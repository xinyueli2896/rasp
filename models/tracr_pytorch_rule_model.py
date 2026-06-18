from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS

# ─────────────────────────────────────────────────────────────────────────────
# RASP program compiled to a transformer (TRACR-style).
#
#   lookup_selector = Select(indices, indices, lambda k, q: k == (q+1) % 4)
#   predicted_next  = Aggregate(lookup_selector, tokens)
#
# Residual stream  d = V*2 + 4 = 28 dims:
#   dims  0 .. V-1    : one-hot current token t_q             (W_E)
#   dims  V .. 2V-1   : one-hot predicted next token          (attention output)
#   dims 2V .. 2V+3   : one-hot position class q%4            (W_pos)
#
# Attention weights (period-4 causal trick):
#   The selector k==(q+1)%4 is non-causal, but t_{q-3}==t_{q+1} for a
#   period-4 sequence, so we attend to k=q-3 instead (causal for q>=3).
#   W_Q shifts pos-class by +1: Q[q] has pos-class (q+1)%4.
#   W_K is identity on pos subspace: K[k] has pos-class k%4.
#   Score(q,k) = 1 iff k%4 == (q+1)%4  →  causal match at k=q-3.
#
#   q >= 3 : sharp attention to k=q-3, dims V..2V-1 = e_{t_{q+1}}  ✓
#   q <  3 : no valid causal key → soft/uniform over available keys
#             (honest uncertainty — standard TRACR behaviour)
#
# No trainable parameters.
# ─────────────────────────────────────────────────────────────────────────────


class TracrPyTorchRuleModel(nn.Module):
    TRACR_D_MODEL: int = VOCAB_SIZE * 2 + 4   # 28

    def __init__(self, max_seq_len: int = 128, attn_scale: float = 20.0):
        super().__init__()
        V = VOCAB_SIZE   # 12
        d = self.TRACR_D_MODEL   # 28
        self.V = V
        self.d_model = d
        self.attn_scale = attn_scale

        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, 2 * V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # W_Q shifts pos-class forward by +1
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[2 * V + (i + 1) % 4, 2 * V + i] = 1.0
        self.register_buffer("W_Q", W_Q)

        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[2 * V + i, 2 * V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V copies token (dims 0..V-1) into next-token slot (V..2V-1)
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[V + i, i] = 1.0
        self.register_buffer("W_V", W_V)

        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[V + i, V + i] = 1.0
        self.register_buffer("W_O", W_O)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  token indices in [0, VOCAB_SIZE).

        Returns logits (B, T, V) and optionally hidden state (B, T, 28).
          q >= 3 : logits are a clean one-hot of the correct next token.
          q <  3 : logits are a soft distribution (honest uncertainty).
        """
        B, T = idx.shape
        t_idx = torch.arange(T, device=idx.device)

        h     = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 28)
        Q     = h @ self.W_Q.t()
        K     = h @ self.W_K.t()
        V     = h @ self.W_V.t()
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale
        mask   = torch.ones(T, T, device=idx.device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        h_out  = h + (attn @ V) @ self.W_O.t()                       # (B, T, 28)
        logits = h_out[:, :, self.V : 2 * self.V]                    # (B, T, V)

        if return_hidden:
            return logits, h_out
        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)
    print("Input    :", inp.squeeze().tolist())
    print("Preds    :", logits.argmax(-1).squeeze().tolist())
    print("Expected : [?, ?, ?, 0, 5, 7, 0, 0]  (q<3 = honest uncertainty)")
