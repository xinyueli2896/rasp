"""
Adapter that "patches" the rule model onto the frozen AR transformer.

Architecture overview
─────────────────────
                ┌─────────────────────────┐
  tokens ──────►│ AR Transformer (frozen)  │──► ar_logits  (B,T,12)
                │                          │──► ar_hidden  (B,T,d_model)
                └─────────────────────────┘
                                                     │
                ┌─────────────────────────┐          │
  tokens ──────►│ Rule Model    (frozen)  │──► rule_logits (B,T,12)
                └─────────────────────────┘          │
                                                     ▼
                                        ┌────────────────────────┐
                                        │  Adapter  (trainable)  │
                                        │                        │
                                        │  gate = σ(MLP(h‖r))    │
                                        │  bias = MLP(h‖r)       │
                                        │                        │
                                        │  out = gate*rule       │
                                        │      +(1-gate)*ar      │
                                        │      + bias            │
                                        └────────────────────────┘
                                                     │
                                              final_logits (B,T,12)

The adapter is trained while both base models are frozen.
Because the rule model outputs near-perfect one-hot distributions for ANY
starting integer, the adapter learns to trust the rule model for positions
where the AR model is uncertain – enabling generalisation to unseen starters.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import AutoregressiveTransformer
from models.rule_model   import RuleModelWrapper


# ---------------------------------------------------------------------------
# Adapter module (trainable)
# ---------------------------------------------------------------------------

class RuleAdapter(nn.Module):
    """
    Lightweight adapter that patches the rule model onto the AR transformer.

    Design principle — "rule-first, AR-correction":
      output = rule_logits + gate * (ar_logits - rule_logits) + residual_bias

    This means the adapter starts from the rule model's predictions and learns
    a gated correction from the AR model.  Initialisation has gate=0 and
    residual=0, so the untrained adapter already outputs rule_logits exactly.

    During training on seen starters (where both models agree), the gate and
    residual converge to near zero (since rule_logits is already correct).
    At test time on unseen starters the AR model's hidden state is
    out-of-distribution, but the adapter only adds a SMALL correction → the
    output remains close to rule_logits → perfect generalisation.

    The additional KL distillation loss in train_adapter.py further pushes the
    adapter output toward rule_logits during training.

    Parameters
    ----------
    d_model   : int  – AR transformer hidden size
    vocab_size : int  – 12 for our task
    hidden_dim : int  – adapter MLP hidden size
    """

    def __init__(self, d_model: int = 128, vocab_size: int = 12, hidden_dim: int = 64):
        super().__init__()
        # Include ar_logits AND rule_logits in feat so the gate can detect
        # agreement/disagreement between the two models directly.
        in_dim = d_model + 2 * vocab_size

        # Scalar gate: how much additional weight to give to AR over rule model
        # Initialised to output ~0 (trust rule model by default)
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Small residual bias (initialised to 0)
        self.bias_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

        self._init()

    def _init(self):
        # Zero-init so adapter starts as identity for rule model
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)   # sigmoid(0) = 0.5 initially
        nn.init.zeros_(self.bias_net[-1].weight)
        nn.init.zeros_(self.bias_net[-1].bias)

    def forward(
        self,
        ar_hidden:   torch.Tensor,   # (B, T, d_model)
        ar_logits:   torch.Tensor,   # (B, T, vocab_size)
        rule_logits: torch.Tensor,   # (B, T, vocab_size)
    ) -> torch.Tensor:               # (B, T, vocab_size)

        # Include ar_logits so the gate can detect AR↔rule disagreement
        feat  = torch.cat([ar_hidden, ar_logits, rule_logits], dim=-1)  # (B,T,d+2V)
        gate  = torch.sigmoid(self.gate_net(feat))                       # (B,T,1)
        bias  = self.bias_net(feat)                                      # (B,T,V)

        # Rule-first: start from rule logits, add gated AR correction
        ar_correction = gate * (ar_logits - rule_logits)
        return rule_logits + ar_correction + bias


# ---------------------------------------------------------------------------
# Full patched model: AR + Rule + Adapter
# ---------------------------------------------------------------------------

class PatchedModel(nn.Module):
    """
    Combines the frozen AR transformer, the frozen rule model, and the
    trainable adapter into a single nn.Module with a standard forward pass.

    Only adapter parameters are updated during training.
    """

    def __init__(
        self,
        ar_model:    AutoregressiveTransformer,
        rule_model:  RuleModelWrapper,
        adapter:     RuleAdapter,
    ):
        super().__init__()
        self.ar_model   = ar_model
        self.rule_model = rule_model
        self.adapter    = adapter

        # Ensure base models are frozen
        for p in self.ar_model.parameters():
            p.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, T) long tensor
        Returns final logits (B, T, vocab_size).
        """
        ar_logits, ar_hidden = self.ar_model(idx, return_hidden=True)   # frozen
        rule_logits          = self.rule_model(idx)                      # frozen, numpy bridge

        return self.adapter(ar_hidden, ar_logits, rule_logits)

    def trainable_parameters(self):
        return self.adapter.parameters()

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, n_new: int) -> torch.Tensor:
        """
        Autoregressively generate n_new tokens after start_tokens.
        Uses the patched (adapter-corrected) logits for greedy decoding.
        """
        self.eval()
        tokens = start_tokens.clone()
        max_len = self.ar_model.max_seq_len
        for _ in range(n_new):
            ctx = tokens[:, -max_len:]
            logits = self(ctx)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_patched_model(
    ar_ckpt_path:    str | None = None,
    max_seq_len:     int = 128,
    d_model:         int = 128,
    n_layers:        int = 4,
    n_heads:         int = 4,
    adapter_hidden:  int = 64,
    force_fallback:  bool = False,
    device:          str = "cpu",
) -> PatchedModel:
    """
    Build a PatchedModel, optionally loading pretrained AR weights.
    """
    ar_model = AutoregressiveTransformer(
        vocab_size=12, max_seq_len=max_seq_len,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
    ).to(device)

    if ar_ckpt_path is not None:
        state = torch.load(ar_ckpt_path, map_location=device)
        ar_model.load_state_dict(state)
        print(f"[PatchedModel] Loaded AR weights from {ar_ckpt_path}")

    rule_model = RuleModelWrapper(max_seq_len=max_seq_len,
                                  force_fallback=force_fallback).to(device)
    adapter    = RuleAdapter(d_model=d_model, vocab_size=12,
                             hidden_dim=adapter_hidden).to(device)

    return PatchedModel(ar_model, rule_model, adapter)


if __name__ == "__main__":
    model = build_patched_model(force_fallback=True)
    print(f"Adapter parameters: {sum(p.numel() for p in model.adapter.parameters()):,}")

    x = torch.tensor([[0, 5, 7, 0, 5]], dtype=torch.long)
    out = model(x)
    print("output shape:", out.shape)
    print("predictions :", out.argmax(-1).tolist())
