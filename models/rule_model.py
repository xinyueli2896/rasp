"""
PyTorch-compatible wrapper around the rule model.

RuleModelWrapper.forward(idx) behaves like the AR transformer:
  idx    : (B, T) long tensor  – input token IDs (0-11)
  returns: (B, T, 12) float tensor  – one-hot logits, part of the autograd graph

Gradient design
───────────────
The rule model is "frozen" – it has no learnable parameters and its weights
never change.  But its output IS part of the autograd computation graph:

  rule_logits = rule_model(idx)   # requires_grad=True leaf tensor

This means:
  • ∂loss/∂rule_logits is computed during backward() and accumulates in
    rule_logits.grad (useful for interpretability / gradient analysis).
  • Gradient flows through rule_logits into the adapter's v_linear and
    out_proj computations even though those layers are also frozen.

Implementation: the forward is pure PyTorch (no numpy, no torch.no_grad),
and the output tensor is detached from integer-arithmetic ops then marked
requires_grad=True so it acts as a differentiable leaf.

The rule
────────
Rule: at sequence position t, the predicted next token is
      (x + OFFSETS[(t+1) % 4]) % 12
where x = idx[:, 0] (starting integer) and OFFSETS = [0, 5, 7, 0].
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS


class RuleModelWrapper(nn.Module):
    """
    Frozen rule model with autograd-compatible output.

    The rule is implemented in pure PyTorch (no numpy / JAX).
    Output tensor has requires_grad=True so it participates in the
    autograd graph without requiring any trainable parameters.

    parameters() returns empty – this module contributes no weights
    to the optimizer.  The "frozen" constraint is enforced by having
    nothing to update, not by disabling requires_grad on the output.
    """

    def __init__(self, max_seq_len: int = 128, force_fallback: bool = False):
        super().__init__()
        # Register OFFSETS as a non-trainable buffer so it moves with the model
        self.register_buffer(
            "_offsets", torch.tensor(OFFSETS, dtype=torch.long)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, T) long tensor
        Returns logits (B, T, VOCAB_SIZE) float tensor.

        The returned tensor is a requires_grad=True leaf: it has no
        connection to upstream computations (the rule is deterministic),
        but gradients will accumulate in .grad during backward().
        Call retain_grad() on the result before backward() if you need
        to inspect rule_logits.grad.
        """
        B, T   = idx.shape
        device = idx.device

        # next_token[b, t] = (x[b] + OFFSETS[(t+1) % 4]) % VOCAB_SIZE
        t      = torch.arange(T, device=device)
        o      = self._offsets[(t + 1) % len(self._offsets)]  # (T,) offsets
        x      = idx[:, 0]                                     # (B,) starters
        next_t = (x[:, None] + o[None, :]) % VOCAB_SIZE        # (B, T) long

        logits = F.one_hot(next_t, num_classes=VOCAB_SIZE).float()  # (B, T, V)

        # Detach from integer arithmetic (which has no grad), then mark as a
        # requires_grad leaf so the tensor participates in the autograd graph.
        return logits.detach().requires_grad_(True)

    def parameters(self, recurse=True):
        return iter([])   # no trainable parameters; excluded from optimizer

    def named_parameters(self, prefix="", recurse=True):
        return iter([])


if __name__ == "__main__":
    model = RuleModelWrapper()
    # x=0: period-4 sequence = 0, 5, 7, 0, 0, 5, 7, 0, ...
    # next-token predictions:   5, 7, 0, 0, 5, 7, 0
    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7]], dtype=torch.long)
    logits = model(inp)
    print("logits shape    :", logits.shape)
    print("requires_grad   :", logits.requires_grad)
    print("grad_fn         :", logits.grad_fn, " (None = leaf tensor, grad in .grad)")
    preds = logits.argmax(-1)
    print("predictions     :", preds.tolist())
    expected = [[5, 7, 0, 0, 5, 7, 0]]
    print("expected        :", expected)
    print("correct         :", preds.tolist() == expected)

    # Verify gradient flows from downstream computation back to rule_logits
    dummy_weight = torch.randn(VOCAB_SIZE, VOCAB_SIZE, requires_grad=True)
    loss = (logits @ dummy_weight).sum()
    loss.backward()
    print("\nGradient test:")
    print("  ∂loss/∂dummy_weight computed:", dummy_weight.grad is not None)
    # Leaf tensors with requires_grad=True accumulate .grad automatically
    print("  rule_logits.grad shape:", logits.grad.shape)
