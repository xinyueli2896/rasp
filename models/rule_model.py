"""
PyTorch-compatible wrapper around the rule model.

RuleModelWrapper.forward(idx, return_hidden=False):
  idx    : (B, T) long tensor  – input token IDs (0-11)
  returns: (B, T, VOCAB_SIZE) logits  (and optionally (B, T, rule_d_model) hidden)

The rule model has its own token embedding table (rule_tok_emb) that is
independent of the AR model's vocabulary.  Hidden states are the embeddings
of the rule's predicted next tokens in this private space:

  rule_hidden[b, t] = rule_tok_emb[predicted_next[b, t]]   # (rule_d_model,)

This decouples the rule model's representation from the AR model: no shared
lm_head.weight projection is needed.

Gradient design
───────────────
The rule model is "frozen" – it has no learnable parameters and its weights
never change.  But its logit output IS part of the autograd computation graph:

  rule_logits = rule_model(idx)   # requires_grad=True leaf tensor

This means:
  • ∂loss/∂rule_logits is computed during backward() and accumulates in
    rule_logits.grad (useful for interpretability / gradient analysis).
  • Gradient flows through rule_logits into the adapter's v_linear and
    out_proj computations even though those layers are also frozen.

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
    Frozen rule model with autograd-compatible output and its own hidden space.

    The rule is implemented in pure PyTorch (no numpy / JAX).
    Logit output has requires_grad=True so it participates in the autograd graph.

    rule_tok_emb is a fixed (non-trainable) (VOCAB_SIZE, rule_d_model) buffer
    that serves as the rule model's private token embedding table, independent
    of the AR model's vocabulary.  Hidden states are embeddings of predicted
    next tokens in this private space.

    parameters() returns empty – this module contributes no weights
    to the optimizer.  The "frozen" constraint is enforced by having
    nothing to update, not by disabling requires_grad on the output.
    """

    def __init__(self, max_seq_len: int = 128, rule_d_model: int = 128,
                 force_fallback: bool = False):
        super().__init__()
        self.rule_d_model = rule_d_model
        # Register OFFSETS as a non-trainable buffer so it moves with the model
        self.register_buffer(
            "_offsets", torch.tensor(OFFSETS, dtype=torch.long)
        )
        # Rule model's own token embeddings — independent of AR model.
        # Initialized as a random matrix; fixed for the lifetime of the model.
        # Saved in state_dict so oracle init is consistent across checkpoints.
        rule_emb = torch.randn(VOCAB_SIZE, rule_d_model)
        self.register_buffer("rule_tok_emb", rule_emb)

    def forward(self, idx: torch.Tensor,
                return_hidden: bool = False):
        """
        idx : (B, T) long tensor
        return_hidden : if True, returns (logits, hidden) where
                        hidden (B, T, rule_d_model) = rule_tok_emb[predicted_next].

        logits is a requires_grad=True leaf: gradients accumulate in .grad
        during backward().  Call retain_grad() before backward() to inspect.
        hidden is a plain tensor slice of the fixed rule_tok_emb buffer.
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
        logits = logits.detach().requires_grad_(True)

        if return_hidden:
            # rule_hidden[b, t] = rule_tok_emb[next_t[b, t]]  (B, T, rule_d_model)
            hidden = self.rule_tok_emb[next_t]
            return logits, hidden
        return logits

    def parameters(self, recurse=True):
        return iter([])   # no trainable parameters; excluded from optimizer

    def named_parameters(self, prefix="", recurse=True):
        return iter([])


if __name__ == "__main__":
    model = RuleModelWrapper(rule_d_model=128)
    # x=0: period-4 sequence = 0, 5, 7, 0, 0, 5, 7, 0, ...
    # next-token predictions:   5, 7, 0, 0, 5, 7, 0
    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)
    print("logits shape    :", logits.shape)
    print("hidden shape    :", hidden.shape)
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
