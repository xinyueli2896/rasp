from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS
from models.tracr_pytorch_rule_model import TracrPyTorchRuleModel


class RuleModelWrapper(nn.Module):
    # Wraps the rule transformer (period-4 causal attention + MLP patch for q<3).
    # forward(idx, return_hidden=True) returns (logits, rule_hidden) where
    # rule_hidden is a 28-dim hidden state:
    #   dims  0-11 : one-hot of current token at position q
    #   dims 12-23 : one-hot of next token (all positions)
    #   dims 24-27 : one-hot of q % 4
    # parameters() returns empty — no trainable parameters.

    def __init__(self, max_seq_len: int = 128,
                 rule_d_model: int = None,   # ignored, always TRACR_D_MODEL
                 force_fallback: bool = False):
        super().__init__()
        self._tracr = TracrPyTorchRuleModel(max_seq_len=max_seq_len)
        self.rule_d_model = self._tracr.TRACR_D_MODEL   # 28

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        return self._tracr(idx, return_hidden=return_hidden)

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
    print("expected(q>=3):", "positions 3-7 → [0, 5, 7, 0, 0]")
