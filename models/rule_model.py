from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS


class RuleModelWrapper(nn.Module):
    # Frozen analytical rule model.
    # rule_d_model = VOCAB_SIZE = 12.
    # forward(idx, return_hidden=True) returns (logits, rule_hidden) where
    # rule_hidden[q] = one_hot(next_token[q]) shape (B, T, 12) — the exact
    # next-token distribution at every position, matching teacher-forced AR targets.
    # parameters() returns empty — no trainable parameters.

    def __init__(self, max_seq_len: int = 128,
                 rule_d_model: int = None,   # ignored, always VOCAB_SIZE
                 force_fallback: bool = False):
        super().__init__()
        self.rule_d_model = VOCAB_SIZE   # 12
        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        B, T = idx.shape
        device = idx.device
        t_idx = torch.arange(T, device=device)
        key = idx[:, 0]   # starter token determines the whole sequence
        # next_token[q] = (key + OFFSETS[(q+1)%4]) % VOCAB_SIZE
        next_t = (key[:, None] + self._offsets[(t_idx + 1) % 4][None, :]) % VOCAB_SIZE
        logits = F.one_hot(next_t, num_classes=VOCAB_SIZE).float()   # (B, T, 12)
        if return_hidden:
            return logits, logits   # rule_hidden = next-token distribution
        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = RuleModelWrapper()
    print(f"rule_d_model = {model.rule_d_model}")

    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)
    print("logits shape :", logits.shape)
    print("hidden shape :", hidden.shape)
    print("predictions  :", logits.argmax(-1).tolist())
    print("expected     :", [[5, 7, 0, 0, 5, 7, 0, 0]])
