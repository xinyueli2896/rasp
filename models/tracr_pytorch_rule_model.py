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
# Attention (period-4 causal trick):
#   W_Q shifts pos-class by +1, so Q[q] targets K[k] where k%4 == (q+1)%4.
#   The causal equivalent is k = q-3, valid for q >= 3.
#   → For q >= 3: dims V..2V-1 get e_{t_{q+1}} (exact next token).
#   → For q <  3: no valid causal key → noisy fallback.
#
# MLP patch (q < 3 only):
#   Local computation: next_token = (t_q + delta[q%4]) % V
#   where delta[p] = (OFFSETS[(p+1)%4] - OFFSETS[p]) % V = [5, 2, 5, 0].
#   Implemented as bilinear lookup W_local: outer(e_{t_q}, e_{q%4}) → e_{next_token}.
#   Uses only dims 0..V-1 and 2V..2V+3 — strictly local, no future access.
#   Overwrites the noisy attention output in dims V..2V-1 for q < 3.
#
# Result: dims V..2V-1 = e_{next_token} for ALL q.
#   q >= 3: from attention (cross-position, non-trivial).
#   q <  3: from MLP patch (local, exact).
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

        # ── Embedding matrices ────────────────────────────────────────────
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, 2 * V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # ── Attention weight matrices (period-4 causal trick) ─────────────
        # W_Q shifts pos-class forward by +1:
        #   Q[q] has pos-class (q+1)%4 → Score(q,k)=1 iff k%4==(q+1)%4 → k=q-3
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[2 * V + (i + 1) % 4, 2 * V + i] = 1.0
        self.register_buffer("W_Q", W_Q)

        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[2 * V + i, 2 * V + i] = 1.0
        self.register_buffer("W_K", W_K)

        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[V + i, i] = 1.0
        self.register_buffer("W_V", W_V)

        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[V + i, V + i] = 1.0
        self.register_buffer("W_O", W_O)

        # ── MLP patch: local bilinear lookup for q < 3 ───────────────────
        # delta[p] = (OFFSETS[(p+1)%4] - OFFSETS[p]) % V
        # W_local[t*4 + p] = e_{(t + delta[p]) % V}
        offsets = torch.tensor(OFFSETS, dtype=torch.long)
        tok_idx = torch.arange(V, dtype=torch.long)
        pos_idx = torch.arange(4, dtype=torch.long)
        delta   = (offsets[(pos_idx + 1) % 4] - offsets[pos_idx]) % V   # [5,2,5,0]
        next_tok = (tok_idx[:, None] + delta[None, :]) % V               # (V, 4)
        W_local  = F.one_hot(next_tok.reshape(-1), V).float()            # (V*4, V)
        self.register_buffer("W_local", W_local)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  — token indices in [0, VOCAB_SIZE).
        Returns logits (B, T, V) and optionally hidden state (B, T, 28).

        dims V..2V-1 of h_out:
          q >= 3 : e_{t_{q+1}} from period-4 attention (cross-position, causal).
          q <  3 : e_{next_token} from local MLP patch (overwrites noisy attn).
        """
        B, T = idx.shape
        t_idx = torch.arange(T, device=idx.device)

        # Embed: h[q] = W_E[t_q] + W_pos[q%4]
        h = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 28)

        # ── Attention (runs for all positions) ────────────────────────────
        Q = h @ self.W_Q.t()
        K = h @ self.W_K.t()
        V = h @ self.W_V.t()

        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale
        mask   = torch.ones(T, T, device=idx.device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)

        attn_out = (attn @ V) @ self.W_O.t()
        h_out    = h + attn_out                                    # (B, T, 28)

        # ── MLP patch for q < 3 ──────────────────────────────────────────
        # Attention has no valid causal key for q < 3 → overwrite dims V..2V-1
        # with exact next token computed from local info (dims 0..V-1, 2V..2V+3).
        cold = t_idx < 3                                           # (T,)
        if cold.any():
            cur   = h[:, cold, :self.V]                            # (B, n, V)
            pos   = h[:, cold, 2 * self.V:]                        # (B, n, 4)
            outer = cur.unsqueeze(-1) * pos.unsqueeze(-2)          # (B, n, V, 4)
            e_nxt = outer.reshape(B, -1, self.V * 4) @ self.W_local  # (B, n, V)
            h_out = h_out.clone()
            h_out[:, cold, self.V : 2 * self.V] = e_nxt

        logits = h_out[:, :, self.V : 2 * self.V]                 # (B, T, V)

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

    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)
    preds    = logits.argmax(-1).squeeze().tolist()
    expected = [5, 7, 0, 0, 5, 7, 0, 0]

    print(f"\nInput    : {inp.squeeze().tolist()}")
    print(f"Preds    : {preds}")
    print(f"Expected : {expected}")
    print(f"All OK   : {preds == expected}")

    print(f"\nHidden state (28-dim):")
    for q in range(8):
        hq   = hidden[0, q]
        cur  = hq[:12].argmax().item()
        nxt  = hq[12:24].argmax().item()
        pos  = hq[24:].argmax().item()
        src  = "MLP" if q < 3 else "attn"
        print(f"  q={q}: cur={cur}  next={nxt}  pos%4={pos}  [{src}]")

    from rasp_program.sequence_rule import OFFSETS
    print(f"\nUnseen starter x=6:")
    x   = 6
    seq = [(x + OFFSETS[i % 4]) % 12 for i in range(8)]
    exp = [(x + OFFSETS[(i + 1) % 4]) % 12 for i in range(8)]
    p2  = model(torch.tensor([seq], dtype=torch.long)).argmax(-1).squeeze().tolist()
    print(f"  Input   : {seq}")
    print(f"  Expected: {exp}")
    print(f"  Preds   : {p2}")
    print(f"  All OK  : {p2 == exp}")
