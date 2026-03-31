"""
PyTorch-compatible wrapper around the rule model.

RuleModelWrapper.forward(idx, return_hidden=False):
  idx    : (B, T) long tensor  – input token IDs
  returns: (B, T, VOCAB_SIZE) logits  (and optionally (B, T, rule_d_model) hidden)

Hidden states
─────────────
Hidden states come from TracrPyTorchRuleModel — a PyTorch transformer whose
weights are analytically constructed to be equivalent to what Tracr compiles
from the RASP program.  They are the TRUE residual stream after the single
attention layer:

  hidden[b, t] = h_out[t]  = [e_{token[t]} + e_{token[(t+1)%4]},  e_{t%4}]

where the first 24 dims carry the current-token one-hot PLUS the predicted-
next-token one-hot, and the last 4 dims carry the period-4 position one-hot.

This replaces the previous fixed random embedding table.

rule_d_model
────────────
rule_d_model is always TracrPyTorchRuleModel.TRACR_D_MODEL = VOCAB_SIZE + 4 = 28.
The YinyangCrossAttention k_proj/v_proj therefore project from 28 → adapter_rank.

Gradient design
───────────────
The rule model has no trainable parameters and its logit output IS part of
the autograd graph (requires_grad=True leaf tensor).

The rule
────────
Rule: at sequence position t, the predicted next token is
      (x + OFFSETS[(t+1) % 4]) % VOCAB_SIZE
where x = idx[:, 0] (starting integer) and OFFSETS = [0, 5, 7, 0].
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.tracr_pytorch_rule_model import TracrPyTorchRuleModel
from rasp_program.sequence_rule import VOCAB_SIZE


class RuleModelWrapper(nn.Module):
    """
    Frozen rule model with true transformer hidden states.

    Wraps TracrPyTorchRuleModel to provide the same API as the previous
    RuleModelWrapper.  Hidden states are the actual residual stream of the
    analytically-constructed Tracr-equivalent PyTorch transformer, not a
    fixed random embedding table.

    rule_d_model is always 28 (= VOCAB_SIZE + 4 = TracrPyTorchRuleModel.TRACR_D_MODEL).

    parameters() returns empty — no trainable parameters; excluded from optimizer.
    """

    def __init__(self, max_seq_len: int = 128,
                 rule_d_model: int = None,   # kept for API compat, ignored (always 28)
                 force_fallback: bool = False):
        super().__init__()
        self._tracr = TracrPyTorchRuleModel(max_seq_len=max_seq_len)
        self.rule_d_model = self._tracr.TRACR_D_MODEL   # 28

        # OFFSETS buffer for the analytical logit computation (inside TracrPyTorchRuleModel)
        # Kept here so existing code that accesses self._offsets still works
        from rasp_program.sequence_rule import OFFSETS
        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T) long tensor
        return_hidden : if True, returns (logits, hidden) where
                        hidden (B, T, 28) = true residual stream from
                        the Tracr-equivalent PyTorch transformer.

        logits is a requires_grad=True leaf: gradients accumulate in .grad
        during backward().
        """
        return self._tracr(idx, return_hidden=return_hidden)

    def parameters(self, recurse=True):
        return iter([])   # no trainable parameters; excluded from optimizer

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = RuleModelWrapper()
    print(f"rule_d_model = {model.rule_d_model}")

    # x=0: period-4 sequence = 0, 5, 7, 0, 0, 5, 7, 0, ...
    # next-token predictions:   5, 7, 0, 0, 5, 7, 0, 0
    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)
    print("logits shape    :", logits.shape)    # (1, 8, 24)
    print("hidden shape    :", hidden.shape)    # (1, 8, 28)
    print("requires_grad   :", logits.requires_grad)
    preds = logits.argmax(-1)
    print("predictions     :", preds.tolist())
    print("expected        :", [[5, 7, 0, 0, 5, 7, 0, 0]])
    print("correct         :", preds.tolist() == [[5, 7, 0, 0, 5, 7, 0, 0]])

    # Hidden state: q=0 should have 1 at token dim 0 (current) and 5 (next)
    q0 = hidden[0, 0]
    print("\nq=0 token dims  (expect 1 at 0 and 5):", q0[:24].tolist())
    print("q=0 pos   dims  (expect 1 at pos 24)  :", q0[24:].tolist())

    # Gradient flows from downstream computation back to rule_logits
    dummy_weight = torch.randn(VOCAB_SIZE, VOCAB_SIZE, requires_grad=True)
    loss = (logits @ dummy_weight).sum()
    loss.backward()
    print("\nGradient test:")
    print("  ∂loss/∂dummy_weight computed:", dummy_weight.grad is not None)
    print("  rule_logits.grad shape:", logits.grad.shape)
