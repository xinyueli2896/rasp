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
# For a period-4 sequence  tokens[q] = (starter + OFFSETS[q%4]) % V:
#   predicted_next[q] = tokens[(q+1)%4]   ← the token at cycle-position (q+1)%4
#
# TRACR compilation (one attention head, no MLP):
#
#   Residual stream layout  d = V*2 + 4 = 28 dims:
#     dims  0 .. V-1    : one-hot of current token t_q        (from W_E)
#     dims  V .. 2V-1   : one-hot of attended token           (written by attention)
#     dims 2V .. 2V+3   : one-hot of position class q%4       (from W_pos)
#
#   Period-4 causal trick:
#     The selector k==(q+1)%4 is non-causal, but because the sequence has
#     period 4, t_{q-3} == t_{q+1}  (since (q-3)%4 == (q+1)%4).
#     We therefore attend to k = q-3, which is causal for q >= 3.
#
#     For q < 3 there is no valid causal k with the right pos-class, so
#     softmax is uniform over the available keys (honest uncertainty).
#     The rule signal at q < 3 is noisy/incorrect; the adapter handles this
#     via its gate, and generation from prompts of length >= 4 avoids it.
#
#   W_Q shifts pos subspace forward by +1:
#     Q[q] has pos-class (q+1)%4  →  matches K[k] where k%4==(q+1)%4  →  k=q-3
#   W_K identity on pos subspace.
#   W_V copies token (dims 0..V-1) into next-token slot (V..2V-1).
#   W_O identity on next-token slot.
#
#   After residual add (q >= 3):
#     h_out[q] = [e_{t_q} | e_{t_{q+1}} | e_{q%4}]
#   Logits are read directly from dims V..2V-1 of h_out.
#
#   No trainable parameters.
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

        # W_Q: shifts pos subspace forward by +1
        #   Q[q] has pos-class (q+1)%4  →  attends to k where k%4==(q+1)%4  →  k=q-3
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

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  — token indices in [0, VOCAB_SIZE).
        Returns logits (B, T, V) and optionally the full hidden state (B, T, 28).

        logits[b, q] is the predicted next-token distribution for position q,
        read directly from dims V..2V-1 of the residual stream after attention.
        For q >= 3 this is a clean one-hot (attends to k=q-3, same cycle position).
        For q < 3 it is a soft/noisy distribution (no valid causal key).
        """
        B, T = idx.shape
        t_idx = torch.arange(T, device=idx.device)

        # Embed: h[q] = W_E[t_q] + W_pos[q%4]
        h = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 28)

        # Queries, Keys, Values
        Q = h @ self.W_Q.t()   # pos-class (q+1)%4 in dims 2V..2V+3
        K = h @ self.W_K.t()   # pos-class q%4     in dims 2V..2V+3
        V = h @ self.W_V.t()   # token one-hot     in dims V..2V-1

        # Causal attention scores (only pos dims contribute to dot-product)
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale   # (B, T, T)
        mask = torch.ones(T, T, device=idx.device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)                        # (B, T, T)

        # Write attended token into next-token slot, add to residual
        attn_out = (attn @ V) @ self.W_O.t()                   # (B, T, 28)
        h_out = h + attn_out                                    # (B, T, 28)

        # Logits from next-token slot of residual stream
        logits = h_out[:, :, self.V : 2 * self.V]              # (B, T, 12)

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

    print(f"\nInput  : {inp.squeeze().tolist()}")
    print(f"Preds  : {logits.argmax(-1).squeeze().tolist()}")
    print(f"Expected (q>=3): [0, 5, 7, 0]  at positions 3-6 → next tokens are [5,7,0,0]")

    print(f"\nHidden state breakdown (28-dim):")
    for q in range(8):
        h = hidden[0, q]
        cur  = h[:12].argmax().item()
        nxt  = h[12:24].argmax().item()
        pos  = h[24:].argmax().item()
        note = "(fallback: q<3)" if q < 3 else ""
        print(f"  q={q}: cur={cur}  attended={nxt}  pos%4={pos}  {note}")

    tokens = inp.squeeze().tolist()
    next_tokens = logits.argmax(-1).squeeze().tolist()
    print(f"\nVerification (q>=3):")
    all_ok = True
    for q in range(3, 8):
        pred = next_tokens[q]
        exp  = tokens[(q + 1) % 4]   # period-4: next token = token at (q+1)%4
        ok   = pred == exp
        all_ok = all_ok and ok
        tag  = "" if ok else " FAIL"
        print(f"  q={q}: pred={pred} (exp {exp}){tag}")
    print(f"  All correct (q>=3): {all_ok}")
