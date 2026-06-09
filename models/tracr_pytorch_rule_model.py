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
    # 1 attention head, no MLP, no causal mask, no trainable parameters

    TRACR_D_MODEL: int = VOCAB_SIZE + 4   # 16

    def __init__(self, max_seq_len: int = 128, attn_scale: float = 20.0):
        super().__init__()
        d = self.TRACR_D_MODEL
        V = VOCAB_SIZE

        self.d_model    = d
        self.attn_scale = attn_scale

        # W_E[t] = [e_t (24-dim one-hot), 0_4]
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # W_pos[p] = [0_24, e_{p%4} (4-dim one-hot)]
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # W_Q: shifts position subspace by +1 mod 4
        # Q[q] = e_{(q+1)%4} in dims 24-27
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[V + i, V + (i - 1) % 4] = 1.0
        self.register_buffer("W_Q", W_Q)

        # W_K: identity on position subspace
        # K[k] = e_{k%4} in dims 24-27
        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[V + i, V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V: identity on token subspace
        # V[k] = e_{token[k]} in dims 0-23
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[i, i] = 1.0
        self.register_buffer("W_V", W_V)

        # W_O: identity on token subspace
        # maps attn output back into residual token subspace
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[i, i] = 1.0
        self.register_buffer("W_O", W_O)

        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        # idx: (B, T)
        # returns logits (B, T, VOCAB_SIZE) and optionally hidden (B, T, 28)
        # hidden = h_out[q] = [e_{cur_token[q]}, e_{q%4}]
        #
        # CURRENT encoding: both token and position dims are at t, not t+1.
        # The causal mask in YinyangCrossAttention blocks rule_hidden[t+1], so
        # NEXT encoding creates a pathological local minimum where the adapter
        # collapses to attending rule_hidden[1] (= embed(x+7)) for all positions,
        # yielding 0.25 accuracy on unseen starters. With CURRENT encoding the
        # adapter learns to attend rule_hidden[0] (= embed(starter x)) and uses
        # query position encoding to derive the next token — routing that
        # generalises to unseen starters.
        #
        # Hidden states are computed analytically (not via attention) so they
        # are correct for any T, including T < 4 during autoregressive generation.
        B, T   = idx.shape
        device = idx.device

        # Compute current tokens analytically
        t_idx  = torch.arange(T, device=device)
        o      = self._offsets[t_idx % 4]
        x      = idx[:, 0]
        cur_t  = (x[:, None] + o[None, :]) % VOCAB_SIZE   # (B, T)

        # Logits: exact one-hot of current token
        logits = F.one_hot(cur_t, num_classes=VOCAB_SIZE).float()
        logits = logits.detach().requires_grad_(True)

        if return_hidden:
            # h_out[q] = [e_{cur_token[q]}, e_{q%4}]
            # Token subspace (dims 0-23): one-hot of current token
            tok_cur = self.W_E[cur_t]       # (B, T, 28) — e_{cur_token[q]} in dims 0-23
            # Position subspace (dims 24-27): one-hot of q % 4
            pos_idx = t_idx % 4
            pos_emb = self.W_pos[pos_idx]   # (T, 28) — e_{q%4} in dims 24-27

            h_out = tok_cur + pos_emb.unsqueeze(0)   # (B, T, 28)
            return logits, h_out

        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")

    # vocab=12, OFFSETS=[0,5,7,0]: sequence for x=0 → [0,5,7,0,0,5,7,0,...]
    # CURRENT encoding: logits predict current token (same as input)
    tests = [
        (0,  [0, 5, 7, 0, 0, 5, 7, 0],  [0, 5, 7, 0, 0, 5, 7, 0]),
        (2,  [2, 7, 9, 2, 2, 7, 9, 2],  [2, 7, 9, 2, 2, 7, 9, 2]),
        (9,  [9, 2, 4, 9, 9, 2, 4, 9],  [9, 2, 4, 9, 9, 2, 4, 9]),  # (9+5)%12=2, (9+7)%12=4
    ]

    all_ok = True
    for x, seq, expected in tests:
        inp = torch.tensor([seq], dtype=torch.long)
        logits, hidden = model(inp, return_hidden=True)
        preds = logits.argmax(-1).squeeze().tolist()
        ok = preds == expected
        all_ok = all_ok and ok
        print(f"  x={x:2d}  preds={preds}  ok={ok}")

    print(f"All correct: {all_ok}")
    print(f"hidden shape: {hidden.shape}")
