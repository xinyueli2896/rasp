from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.tracr_pytorch_rule_model import TracrPyTorchRuleModel
from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS


class RuleModelWrapper(nn.Module):
    # Frozen rule model with true transformer hidden states.
    # Wraps TracrPyTorchRuleModel; rule_d_model is always 28.
    # parameters() returns empty — no trainable parameters.

    def __init__(self, max_seq_len: int = 128,
                 rule_d_model: int = None,   # ignored, always 28
                 force_fallback: bool = False):
        super().__init__()
        self._tracr = TracrPyTorchRuleModel(max_seq_len=max_seq_len)
        self.rule_d_model = self._tracr.TRACR_D_MODEL   # 28
        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

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
    print("expected     :", [[5, 7, 0, 0, 5, 7, 0, 0]])
