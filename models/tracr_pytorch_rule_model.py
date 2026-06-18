from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS

# ─────────────────────────────────────────────────────────────────────────────
# RASP program:
#
#   lookup_selector = Select(indices, indices, lambda k, q: k == (q+1) % 4)
#   predicted_next  = Aggregate(lookup_selector, tokens)
#
# Residual stream layout  d = V*2 + 4 = 28 dims:
#   dims  0 .. V-1    : one-hot of current token t_q        (from W_E)
#   dims  V .. 2V-1   : one-hot of next token               (written below)
#   dims 2V .. 2V+3   : one-hot of position class q%4       (from W_pos)
#
# Attention (period-4 causal trick, q >= 3):
#   W_Q shifts pos-class by +1 → Q[q] targets k where k%4 == (q+1)%4 → k = q-3.
#   For q >= 3: sharp causal attention to k=q-3, writes e_{t_{q+1}} to dims V..2V-1.
#   For q <  3: no valid causal key → noisy fallback in dims V..2V-1.
#
# MLP patch (q < 3 only):
#   next_token = (t_q + delta[q%4]) % V,  delta = [5, 2, 5, 0]
#   Bilinear lookup W_local: outer(e_{t_q}, e_{q%4}) → e_{next_token}.
#   Overwrites the noisy attention output in dims V..2V-1 for q < 3.
#   Uses only local dims 0..V-1 and 2V..2V+3 — strictly causal.
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

        # W_E[t] = e_t in dims 0..V-1
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # W_pos[p] = e_p in dims 2V..2V+3
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, 2 * V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # W_Q: shifts pos-class forward by +1
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[2 * V + (i + 1) % 4, 2 * V + i] = 1.0
        self.register_buffer("W_Q", W_Q)

        # W_K: identity on pos subspace
        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[2 * V + i, 2 * V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V: copies token dims 0..V-1 into next-token slot V..2V-1
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[V + i, i] = 1.0
        self.register_buffer("W_V", W_V)

        # W_O: identity on next-token slot V..2V-1
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[V + i, V + i] = 1.0
        self.register_buffer("W_O", W_O)

        # W_local: bilinear MLP patch for q < 3
        # W_local[t*4 + p] = e_{(t + delta[p]) % V},  delta = [5,2,5,0]
        offsets = torch.tensor(OFFSETS, dtype=torch.long)
        tok_idx = torch.arange(V, dtype=torch.long)
        pos_idx = torch.arange(4, dtype=torch.long)
        delta   = (offsets[(pos_idx + 1) % 4] - offsets[pos_idx]) % V
        next_tok = (tok_idx[:, None] + delta[None, :]) % V
        self.register_buffer("W_local", F.one_hot(next_tok.reshape(-1), V).float())

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  — token indices in [0, VOCAB_SIZE).
        Returns logits (B, T, V) and optionally hidden state (B, T, 28).

        dims V..2V-1 of h_out:
          q >= 3 : e_{t_{q+1}} from period-4 causal attention.
          q <  3 : e_{next_token} from local bilinear MLP patch.
        """
        B, T = idx.shape
        t_idx = torch.arange(T, device=idx.device)

        h = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 28)

        # Attention
        Q = h @ self.W_Q.t()
        K = h @ self.W_K.t()
        V = h @ self.W_V.t()
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale
        mask   = torch.ones(T, T, device=idx.device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        h_out  = h + (attn @ V) @ self.W_O.t()                   # (B, T, 28)

        # MLP patch for q < 3
        cold = t_idx < 3
        if cold.any():
            cur   = h[:, cold, :self.V]
            pos   = h[:, cold, 2 * self.V:]
            outer = cur.unsqueeze(-1) * pos.unsqueeze(-2)
            e_nxt = outer.reshape(B, -1, self.V * 4) @ self.W_local
            h_out = h_out.clone()
            h_out[:, cold, self.V : 2 * self.V] = e_nxt

        logits = h_out[:, :, self.V : 2 * self.V]

        if return_hidden:
            return logits, h_out
        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")

    inp      = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)
    preds    = logits.argmax(-1).squeeze().tolist()
    expected = [5, 7, 0, 0, 5, 7, 0, 0]
    print(f"Preds   : {preds}")
    print(f"Expected: {expected}  all_ok={preds == expected}")
