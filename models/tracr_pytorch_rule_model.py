from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS


class TracrPyTorchRuleModel(nn.Module):
    # d_model = VOCAB_SIZE * 2 + 4 = 28
    #
    # Hidden state layout (28 dims):
    #   dims  0-11 : one-hot of CURRENT token at position q   (from W_E, residual)
    #   dims 12-23 : one-hot of PREVIOUS token at position q  (written by W_V/W_O via attention)
    #   dims 24-27 : one-hot of q % 4                         (from W_pos, residual)
    #
    # Attention: position q attends to position q-1.
    #   W_Q shifts pos subspace back by 1 → query at q has pos_class (q-1)%4
    #   W_K is identity on pos subspace   → key   at k has pos_class k%4
    #   dot-product scores peak at k = q-1 (unique past k where k%4 == (q-1)%4)
    #   W_V maps current-token dims (0-11) → prev-token dims (12-23)
    #   W_O is identity on dims 12-23
    #
    # After residual add:
    #   h_out[q] = [e_{t_q} ; e_{t_{q-1}} ; e_{q%4}]
    #
    # At q=0 there is no q-1: the causal softmax attends to position 0 itself
    # (score = 0, but it is the only legal key), so h_out[0] = [e_{t_0}; e_{t_0}; e_0].
    #
    # Logits are computed analytically (next-token prediction cannot be expressed
    # as a single linear map on h_out without an MLP).
    #
    # No trainable parameters.

    TRACR_D_MODEL: int = VOCAB_SIZE * 2 + 4   # 28

    def __init__(self, max_seq_len: int = 128, attn_scale: float = 20.0):
        super().__init__()
        V = VOCAB_SIZE   # 12
        d = self.TRACR_D_MODEL   # 28

        self.d_model    = d
        self.attn_scale = attn_scale

        # W_E[t] = e_t in dims 0-11, zeros elsewhere
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # W_pos[p] = e_p in dims 24-27, zeros elsewhere
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, 2 * V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # W_Q: shifts pos subspace (dims 24-27) back by 1 mod 4
        # (h @ W_Q.T)[2V+i] = h[2V+(i+1)%4]  →  Q has pos_class (q-1)%4
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[2 * V + i, 2 * V + (i + 1) % 4] = 1.0
        self.register_buffer("W_Q", W_Q)

        # W_K: identity on pos subspace (dims 24-27)
        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[2 * V + i, 2 * V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V: maps current-token dims (0-11) → prev-token dims (12-23)
        # (h @ W_V.T)[V+i] = h[i]  →  V has e_{t_q} in dims 12-23
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[V + i, i] = 1.0
        self.register_buffer("W_V", W_V)

        # W_O: identity on prev-token dims (12-23)
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[V + i, V + i] = 1.0
        self.register_buffer("W_O", W_O)

        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  token indices.
        Returns logits (B, T, V) and optionally hidden (B, T, 28).

        Logits predict next_token[q] = (x + OFFSETS[(q+1)%4]) % V  (analytical).
        Hidden is computed by actual attention using W_Q/K/V/O:
          h_out[q] = [e_{t_q} ; e_{t_{q-1}} ; e_{q%4}]
        """
        B, T   = idx.shape
        device = idx.device
        t_idx  = torch.arange(T, device=device)
        x      = idx[:, 0]

        # Analytical next-token logits
        next_t = (x[:, None] + self._offsets[(t_idx + 1) % 4][None, :]) % VOCAB_SIZE
        logits = F.one_hot(next_t, num_classes=VOCAB_SIZE).float()
        logits = logits.detach().requires_grad_(True)

        if not return_hidden:
            return logits

        # --- Actual attention forward ---
        # Embed: h[q] = W_E[t_q] + W_pos[q%4]
        h = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 28)

        # Queries, Keys, Values
        Q = h @ self.W_Q.t()    # (B, T, 28): e_{(q-1)%4} in dims 24-27
        K = h @ self.W_K.t()    # (B, T, 28): e_{q%4}     in dims 24-27
        V = h @ self.W_V.t()    # (B, T, 28): e_{t_q}     in dims 12-23

        # Causal attention scores  (only pos dims contribute to dot-product)
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale      # (B, T, T)
        mask   = torch.ones(T, T, device=device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)                         # (B, T, T)

        # Attended value + output projection
        attn_out = (attn @ V) @ self.W_O.t()                      # (B, T, 28)

        # Residual: h_out[q] = [e_{t_q}(0-11) ; e_{t_{q-1}}(12-23) ; e_{q%4}(24-27)]
        h_out = h + attn_out                                       # (B, T, 28)

        return logits, h_out

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")
    print(f"W_E   shape: {model.W_E.shape}")
    print(f"W_pos shape: {model.W_pos.shape}")
    print(f"W_Q   shape: {model.W_Q.shape}")

    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)

    print(f"\nInput : {inp.squeeze().tolist()}")
    print(f"Logits (next-token predictions): {logits.argmax(-1).squeeze().tolist()}")
    print(f"Expected                       : [5, 7, 0, 0, 5, 7, 0, 0]")

    print(f"\nHidden state breakdown (28-dim):")
    for q in range(8):
        h = hidden[0, q]
        cur  = h[:12].argmax().item()
        prev = h[12:24].argmax().item()
        pos  = h[24:].argmax().item()
        print(f"  q={q}: cur={cur}  prev={prev}  pos%4={pos}")

    # Verify attention: cur[q] should equal inp[q], prev[q] should equal inp[q-1]
    tokens = inp.squeeze().tolist()
    print(f"\nVerification:")
    all_ok = True
    for q in range(8):
        h = hidden[0, q]
        cur  = h[:12].argmax().item()
        prev = h[12:24].argmax().item()
        pos  = h[24:].argmax().item()
        exp_cur  = tokens[q]
        exp_prev = tokens[q - 1] if q > 0 else tokens[0]   # q=0: self-reference
        exp_pos  = q % 4
        ok = (cur == exp_cur) and (prev == exp_prev) and (pos == exp_pos)
        all_ok = all_ok and ok
        if not ok:
            print(f"  q={q} FAIL: got cur={cur} prev={prev} pos={pos}, "
                  f"expected cur={exp_cur} prev={exp_prev} pos={exp_pos}")
    print(f"  All correct: {all_ok}")
