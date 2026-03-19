"""
PyTorch-compatible wrapper around the rule model (tracr or fallback).

RuleModelWrapper.forward(idx) behaves like the AR transformer:
  idx    : (B, T) long tensor  – input token IDs (0-11)
  returns: (B, T, 12) float tensor  – logits (one-hot-like)

The underlying computation is the FallbackRuleModel / TracrRuleModel;
numpy arrays are converted to/from torch as needed.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from rasp_program.sequence_rule import build_rule_model, FallbackRuleModel, VOCAB_SIZE


class RuleModelWrapper(nn.Module):
    """
    Frozen, non-trainable wrapper that exposes the rule model with a
    standard nn.Module interface so it can sit next to the AR transformer.

    No parameters – the rule is hard-coded / compiled.
    """

    def __init__(self, max_seq_len: int = 128, force_fallback: bool = False):
        super().__init__()
        self._rule_model = build_rule_model(max_seq_len=max_seq_len,
                                            force_fallback=force_fallback)
        # Mark as frozen (no parameters to train)
        self._is_rule_model = True

    @torch.no_grad()
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, T) long tensor
        Returns logits (B, T, 12) float tensor on the same device as idx.
        """
        device = idx.device
        tokens_np = idx.cpu().numpy().astype(np.int32)
        logits_np = self._rule_model.get_logits_vectorized(tokens_np) \
                    if isinstance(self._rule_model, FallbackRuleModel) \
                    else self._rule_model.get_logits(tokens_np)
        return torch.from_numpy(logits_np).to(device)

    def parameters(self, recurse=True):
        # No parameters; return empty iterator
        return iter([])

    def named_parameters(self, prefix="", recurse=True):
        return iter([])


if __name__ == "__main__":
    model = RuleModelWrapper(force_fallback=True)
    # x=0: sequence = 0,5,7,0,5,7,...
    inp = torch.tensor([[0, 5, 7, 0, 5, 7, 0]], dtype=torch.long)
    logits = model(inp)
    print("logits shape:", logits.shape)
    preds = logits.argmax(-1)
    print("predictions :", preds.tolist())
    # At position t, predicted next = (0 + OFFSETS[(t+1)%3]) % 12
    expected = [5, 7, 0, 5, 7, 0, 5]
    print("expected    :", expected)
    print("correct     :", preds[0].tolist() == expected)
