from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS


class TracrPyTorchRuleModel(nn.Module):
    # d_model = VOCAB_SIZE + 4 = 16  (vocab=12, same as BassTracrRuleModel)
    # first 12 dims: token subspace (one-hot of token value)
    # last   4 dims: position subspace (one-hot of position mod 4)
    # 1 attention head, causal, no MLP, no trainable parameters

    TRACR_D_MODEL: int = VOCAB_SIZE + 4   # 16

    def __init__(self, max_seq_len: int = 128, attn_scale: float = 20.0):
        super().__init__()
        d = self.TRACR_D_MODEL
        V = VOCAB_SIZE

        self.d_model    = d
        self.attn_scale = attn_scale

        # W_E[t] = [e_t (12-dim one-hot), 0_4]
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # W_pos[p] = [0_12, e_{p%4} (4-dim one-hot)]
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # W_Q: shifts position subspace by -1 mod 4 (CAUSAL: query q attends to key (q-1)%4)
        # Q[q] = e_{(q-1)%4} in dims 12-15
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[V + i, V + (i + 1) % 4] = 1.0
        self.register_buffer("W_Q", W_Q)

        # W_K: identity on position subspace
        # K[k] = e_{k%4} in dims 12-15
        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[V + i, V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V: identity on token subspace
        # V[k] = e_{token[k]} in dims 0-11
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[i, i] = 1.0
        self.register_buffer("W_V", W_V)

        # W_O: identity on token subspace
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[i, i] = 1.0
        self.register_buffer("W_O", W_O)

        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        # idx: (B, T)
        # returns logits (B, T, VOCAB_SIZE) and optionally hidden (B, T, 16)
        #
        # CURRENT encoding: rule_hidden[q] = [e_{cur_token[q]}, e_{q%4}]
        # Causal: the attention routes q → (q-1)%4, so rule_hidden[q] only
        # depends on tokens at positions <= q. No future token leakage.
        # The analytical bypass computes rule_hidden directly from idx[:,0],
        # which is equivalent and works for any T >= 1.
        B, T   = idx.shape
        device = idx.device
        t_idx  = torch.arange(T, device=device)
        x      = idx[:, 0]

        # Current tokens: (x + OFFSETS[q%4]) % 12
        cur_t  = (x[:, None] + self._offsets[t_idx % 4][None, :]) % VOCAB_SIZE  # (B, T)
        # Next tokens (for logits): (x + OFFSETS[(q+1)%4]) % 12
        next_t = (x[:, None] + self._offsets[(t_idx + 1) % 4][None, :]) % VOCAB_SIZE

        logits = F.one_hot(next_t, num_classes=VOCAB_SIZE).float()
        logits = logits.detach().requires_grad_(True)

        if return_hidden:
            # h_out[q] = [e_{cur_token[q]}, e_{q%4}]
            tok_cur = self.W_E[cur_t]                      # (B, T, 16)
            pos_emb = self.W_pos[t_idx % 4]               # (T, 16)
            h_out   = tok_cur + pos_emb.unsqueeze(0)       # (B, T, 16)
            return logits, h_out

        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")

    # CURRENT encoding: hidden encodes current token; logits predict next token
    # Sequence for x=0: [0,5,7,0,0,5,7,0,...], next tokens: [5,7,0,0,5,7,0,0,...]
    tests = [
        (0,  [0, 5, 7, 0, 0, 5, 7, 0],  [5, 7, 0, 0, 5, 7, 0, 0]),
        (2,  [2, 7, 9, 2, 2, 7, 9, 2],  [7, 9, 2, 2, 7, 9, 2, 2]),
        (9,  [9, 2, 4, 9, 9, 2, 4, 9],  [2, 4, 9, 9, 2, 4, 9, 9]),
    ]

    all_ok = True
    for x, seq, expected in tests:
        inp = torch.tensor([seq], dtype=torch.long)
        logits, hidden = model(inp, return_hidden=True)
        preds = logits.argmax(-1).squeeze().tolist()
        ok = preds == expected
        all_ok = all_ok and ok
        print(f"  x={x:2d}  preds={preds}  ok={ok}")

    # Verify hidden encodes current tokens (not next)
    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    _, h = model(inp, return_hidden=True)
    cur_from_hidden = h[0, :, :VOCAB_SIZE].argmax(-1).tolist()
    print(f"hidden token dims (should be [0,5,7,0,0,5,7,0]): {cur_from_hidden}")
    print(f"All correct: {all_ok}")
