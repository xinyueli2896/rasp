"""
Analytical TracR-style rule model for bar-level chord root prediction.

Exact structural analogue of TracrPyTorchRuleModel, but operating on
chord roots (mod 12) rather than integer tokens (mod 24).

Rule:  chord_root[bar] = (key + OFFSETS[bar % 4]) % 12
       OFFSETS = [0, 5, 7, 0]   (I – IV – V – I)

Input : chord root indices  (B, n_bars)   int64, values in 0-11
Output: logits              (B, n_bars, 12)
        hidden              (B, n_bars, TRACR_D_MODEL=16)   if return_hidden=True

Hidden state format (mirrors integer-sequence TracR):
  dims  0-11 : one-hot of predicted next chord root
  dims 12-15 : one-hot of bar % 4

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

    def forward(self, roots: torch.Tensor, return_hidden: bool = False):
        """
        roots : (B, T)  chord root indices in 0-11
                (extracted from full chord tokens as token // N_QUALITIES)
        """
        B, T   = roots.shape
        device = roots.device

        t_idx  = torch.arange(T, device=device)
        o      = self._offsets[(t_idx + 1) % N_POS]
        key    = roots[:, 0]                                  # starter = first bar root
        next_r = (key[:, None] + o[None, :]) % N_ROOTS       # (B, T)

        logits = F.one_hot(next_r, num_classes=N_ROOTS).float()
        logits = logits.detach().requires_grad_(True)

        if return_hidden:
            tok_next = self.W_E[next_r]                       # (B, T, 16) root dims
            pos_emb  = self.W_pos[t_idx % N_POS]             # (T, 16) pos dims
            h_out    = tok_next + pos_emb.unsqueeze(0)        # (B, T, 16)
            return logits, h_out

        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    from midi_adapter.chord_tokenizer import N_QUALITIES

    model = BassTracrRuleModel()
    print(f'TRACR_D_MODEL = {model.TRACR_D_MODEL}')

    # key=0 (C): roots should be [C,F,G,C,...] = [0,5,7,0,...]
    # predicted NEXT root at each bar: [5,7,0,0, 5,7,0,0]
    roots = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(roots, return_hidden=True)
    preds = logits.argmax(-1).squeeze().tolist()
    print(f'roots  : {roots.squeeze().tolist()}')
    print(f'preds  : {preds}')
    print(f'expect : [5, 7, 0, 0, 5, 7, 0, 0]')
    print(f'hidden shape: {hidden.shape}')
    print(f'All correct: {preds == [5, 7, 0, 0, 5, 7, 0, 0]}')
