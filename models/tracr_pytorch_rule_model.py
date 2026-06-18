from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from rasp_program.sequence_rule import VOCAB_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Rule model hidden state — 16 dims:
#
#   dims  0 .. V-1    : one-hot of current token t_q      (12 dims)
#   dims  V .. V+3    : one-hot of position class q%4     ( 4 dims)
#
# The adapter receives this and learns the rule itself:
#   next_token = f(current_token, pos_class)
#              = (current_token + delta[pos_class]) % V
#              where delta = [5, 2, 5, 0]
#
# No attention, no pre-computed answer — raw ingredients only.
# No trainable parameters.
# ─────────────────────────────────────────────────────────────────────────────


class TracrPyTorchRuleModel(nn.Module):
    TRACR_D_MODEL: int = VOCAB_SIZE + 4   # 16

    def __init__(self, max_seq_len: int = 128, **kwargs):
        super().__init__()
        V = VOCAB_SIZE   # 12
        d = self.TRACR_D_MODEL   # 16

        self.V = V
        self.d_model = d

        # W_E[t] = e_t in dims 0..V-1
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # W_pos[p] = e_p in dims V..V+3
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  — token indices in [0, VOCAB_SIZE).
        Returns logits (B, T, V) — uniform (no rule pre-computed here;
        the adapter learns the rule) — and optionally hidden state (B, T, 16).

        hidden state:
          dims  0..V-1 : e_{t_q}  (current token one-hot)
          dims  V..V+3 : e_{q%4}  (position class one-hot)
        """
        B, T = idx.shape
        t_idx = torch.arange(T, device=idx.device)

        h = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 16)

        # Logits: uniform — the adapter is responsible for predicting next token
        logits = torch.zeros(B, T, self.V, device=idx.device)

        if return_hidden:
            return logits, h
        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")

    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    _, hidden = model(inp, return_hidden=True)

    print(f"\nInput  : {inp.squeeze().tolist()}")
    print(f"\nHidden state (16-dim):")
    for q in range(8):
        h  = hidden[0, q]
        cur = h[:12].argmax().item()
        pos = h[12:].argmax().item()
        print(f"  q={q}: cur_token={cur}  pos%4={pos}")
