"""
Analytical TracR-style rule model for bass pitch-class prediction.

Exact structural analogue of TracrPyTorchRuleModel (integer model, mod 24),
but operating on pitch classes (mod 12) rather than integers (mod 24).

Rule:  sequence[t] = (x + OFFSETS[t % 4]) % 12,   OFFSETS = [0, 5, 7, 0]
       where x = pitch_classes[:, 0]  (starting pitch class, read from sequence)

Input : pitch class indices  (B, T)   int64, values in 0-11
Output: logits               (B, T, 12)
        hidden               (B, T, TRACR_D_MODEL=16)   if return_hidden=True

Hidden state format:
  dims  0-11 : one-hot of CURRENT pitch class at position t  (key + OFFSETS[t%4])
  dims 12-15 : one-hot of t % 4

CURRENT encoding is used (not NEXT) because the causal mask in
CPYinyangCrossAttention blocks rule_hidden[t+1] from the query at t.
With CURRENT encoding the adapter learns to attend rule_hidden[0] (always
accessible, always = embed(key)) and uses the query's positional encoding to
derive the next note — a purely positional routing that generalises to unseen
keys. NEXT encoding causes a pathological local minimum where the model
collapses to attending rule_hidden[1] (= embed(key+7)) for all positions,
yielding 0.25 accuracy on unseen keys.

No trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import OFFSETS

N_ROOTS  = 12
N_POS    = 4
TRACR_D_MODEL = N_ROOTS + N_POS   # 16


class BassTracrRuleModel(nn.Module):

    TRACR_D_MODEL: int = TRACR_D_MODEL

    def __init__(self, max_seq_len: int = 512, attn_scale: float = 20.0):
        super().__init__()
        V = N_ROOTS
        d = TRACR_D_MODEL
        self.attn_scale = attn_scale

        # W_E[r] = [e_r (12-dim one-hot), 0_4]
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer('W_E', W_E)

        # W_pos[p] = [0_12, e_{p%4}]
        W_pos = torch.zeros(N_POS, d)
        for i in range(N_POS):
            W_pos[i, V + i] = 1.0
        self.register_buffer('W_pos', W_pos)

        # W_Q: shifts position subspace by +1 mod 4
        W_Q = torch.zeros(d, d)
        for i in range(N_POS):
            W_Q[V + i, V + (i - 1) % N_POS] = 1.0
        self.register_buffer('W_Q', W_Q)

        # W_K: identity on position subspace
        W_K = torch.zeros(d, d)
        for i in range(N_POS):
            W_K[V + i, V + i] = 1.0
        self.register_buffer('W_K', W_K)

        # W_V: identity on root subspace
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[i, i] = 1.0
        self.register_buffer('W_V', W_V)

        # W_O: identity on root subspace
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[i, i] = 1.0
        self.register_buffer('W_O', W_O)

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  pitch class indices in 0-11; idx[:, 0] determines the key.
        """
        B, T   = idx.shape
        device = idx.device

        t_idx  = torch.arange(T, device=device)
        o      = self._offsets[t_idx % N_POS]                  # CURRENT pitch class at t
        key    = idx[:, 0]                                     # starting pitch class
        cur_r  = (key[:, None] + o[None, :]) % N_ROOTS        # (B, T) current pitch class

        logits = F.one_hot(cur_r, num_classes=N_ROOTS).float()

        if return_hidden:
            tok_cur = self.W_E[cur_r]                          # (B, T, 16) root dims
            pos_emb = self.W_pos[t_idx % N_POS]               # (T, 16) pos dims
            h_out   = tok_cur + pos_emb.unsqueeze(0)           # (B, T, 16)
            return logits, h_out

        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    model = BassTracrRuleModel()
    print(f'TRACR_D_MODEL = {model.TRACR_D_MODEL}')

    # key=0 (C): sequence = [0,5,7,0,0,5,7,0,...]
    # CURRENT pitch class at each position: [0,5,7,0, 0,5,7,0]  (same as input)
    idx = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(idx, return_hidden=True)
    preds = logits.argmax(-1).squeeze().tolist()
    print(f'input  : {idx.squeeze().tolist()}')
    print(f'preds  : {preds}')
    print(f'expect : [0, 5, 7, 0, 0, 5, 7, 0]  (current pitch class)')
    print(f'hidden shape: {hidden.shape}')
    print(f'All correct: {preds == [0, 5, 7, 0, 0, 5, 7, 0]}')
