"""
Low-rank cross-attention adapter that patches the rule model onto the frozen
AR transformer.

Architecture overview
─────────────────────
                ┌─────────────────────────┐
  tokens ──────►│ AR Transformer (frozen)  │──► ar_hidden  (B,T,d_model)
                │                          │──► ar_logits  (B,T,12)
                └─────────────────────────┘
                                                      │
                ┌─────────────────────────┐           │  Q = W_q(ar_hidden)
  tokens ──────►│ Rule Model    (frozen)  │──► rule_logits (B,T,12)
                └─────────────────────────┘           │  K = W_k(rule_logits)
                                                      │  V = rule_logits  (no W_v)
                                                      ▼
                                        ┌─────────────────────────┐
                                        │  Cross-Attention        │
                                        │  (trainable, low-rank)  │
                                        │                         │
                                        │  scores = Q Kᵀ/√r       │
                                        │        + cycle_bias     │
                                        │  attn = softmax(scores  │
                                        │          + causal mask) │
                                        │  attended = attn @ V    │
                                        └─────────────────────────┘
                                                      │
                                 ar_logits + α · attended
                                                      │
                                              final_logits (B,T,12)

The adapter works entirely in *logit space*:
  - Values are rule_logits directly (already in vocab space; no W_v needed)
  - The cross-attention produces an attention-weighted mixture of rule logits
  - A learnable scalar α blends this into ar_logits

"Low-rank" refers to the Q and K projection dimension (rank ≪ d_model).

Cycle-position bias (key design decision)
─────────────────────────────────────────
With purely content-based attention (random W_q, W_k init), the attention
starts approximately uniform and remains so — the optimisation landscape is
flat because mean(rule_logits) ≈ uniform over vocab, making attended ≈ 1/12
everywhere regardless of weights.

The fix: a learnable 4×4 cycle-position bias matrix cycle_bias[c_q, c_k]
(where c_q = t%4 and c_k = s%4) is added to the raw attention scores.
Initialised strongly diagonal (bias=+4 on diagonal), it primes the attention
to attend to same-cycle positions at step 0:

  attended[t] ≈ mean(rule_logits[s] for s where s%4==t%4)
             = rule_logits[t]   (since same-cycle rule logits are identical)

This means the untrained adapter already approximates:
  logits ≈ ar_logits + α · rule_logits[t]

which is the correct oracle blend. W_q and W_k then refine the content-based
component (e.g. adapting to unseen x values), while cycle_bias can also shift
if non-diagonal patterns turn out to be better.

Parameter count (d=128, r=16, cycle_length=4):
  W_q: d×r = 2048,  W_k: 12×r = 192,  cycle_bias: 16,  α: 1  →  2,257 total
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

class LowRankCrossAttentionAdapter(nn.Module):
    """
    Low-rank cross-attention adapter operating in logit space.

    Queries from AR hidden state; keys and values from rule logits.
    The attended output (a mixture of rule one-hot logits) is added to
    ar_logits scaled by a learnable scalar α.

    Parameters
    ----------
    d_model   : int – AR transformer hidden size (query input dimension)
    vocab_size : int – vocabulary size = rule output dimension (12)
    rank      : int – low-rank bottleneck for Q and K projections
    """

    def __init__(
        self,
        d_model:      int = 128,
        vocab_size:   int = 12,
        rank:         int = 16,
        cycle_length: int = 4,     # period of the sequence rule
    ):
        super().__init__()
        self.rank         = rank
        self.scale        = rank ** -0.5
        self.cycle_length = cycle_length

        self.W_q = nn.Linear(d_model,    rank, bias=False)   # AR hidden   → queries
        self.W_k = nn.Linear(vocab_size, rank, bias=False)   # rule logits → keys
        # No W_v: values ARE rule_logits (already in vocab/logit space)

        # cycle_bias[c_q, c_k]: learnable bias added to attention scores
        # when query is at cycle position c_q and key is at cycle position c_k.
        # Initialised strongly diagonal so same-cycle positions attend to each
        # other at step 0, giving attended[t] ≈ rule_logits[t] immediately.
        self.cycle_bias = nn.Parameter(torch.zeros(cycle_length, cycle_length))

        # Blend scalar: α · attended is added to ar_logits.
        # Start at 5.0 so rule signal immediately overcomes typical AR logit gaps
        # (~4 units gap for OOD starters).  Cycle-bias already ensures correct
        # attention from step 0, so there's no need to ramp up slowly.
        self.alpha = nn.Parameter(torch.tensor(5.0))

        self._init()

    def _init(self):
        std = self.rank ** -0.5
        nn.init.normal_(self.W_q.weight, std=std)
        nn.init.normal_(self.W_k.weight, std=std)
        # Diagonal init for cycle_bias: strongly prefer same-cycle attention
        nn.init.constant_(self.cycle_bias, 0.0)
        with torch.no_grad():
            self.cycle_bias.fill_diagonal_(4.0)

    def forward(
        self,
        ar_hidden:   torch.Tensor,   # (B, T, d_model)
        ar_logits:   torch.Tensor,   # (B, T, vocab_size)
        rule_logits: torch.Tensor,   # (B, T, vocab_size)  – one-hot predictions
    ) -> torch.Tensor:               # (B, T, vocab_size)

        B, T, _ = ar_hidden.shape
        device  = ar_hidden.device

        Q = self.W_q(ar_hidden)    # (B, T, rank)
        K = self.W_k(rule_logits)  # (B, T, rank)

        # Content-based scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale          # (B, T, T)

        # Add cycle-position bias: cycle_bias[t%L, s%L] for all (t, s)
        t_idx  = torch.arange(T, device=device)
        bias   = self.cycle_bias[t_idx[:, None] % self.cycle_length,
                                  t_idx[None, :] % self.cycle_length]  # (T, T)
        scores = scores + bias

        # Causal mask
        causal = torch.ones(T, T, device=device).tril().bool()
        scores = scores.masked_fill(~causal, float('-inf'))
        attn   = F.softmax(scores, dim=-1)                        # (B, T, T)

        # Values are rule_logits — output in vocab/logit space
        attended = attn @ rule_logits                             # (B, T, vocab_size)

        return ar_logits + self.alpha * attended


# ---------------------------------------------------------------------------
# Full patched model: AR + Rule + Adapter
# ---------------------------------------------------------------------------

class PatchedModel(nn.Module):
    """
    Combines the frozen AR transformer, the frozen rule model, and the
    trainable cross-attention adapter.

    Forward pass:
      1. ar_hidden, ar_logits = AR transformer   (frozen)
      2. rule_logits           = rule model       (frozen)
      3. logits = adapter(ar_hidden, ar_logits, rule_logits)  (trainable)

    Only adapter parameters are updated during training.
    """

    def __init__(
        self,
        ar_model:    AutoregressiveTransformer,
        rule_model:  RuleModelWrapper,
        adapter:     LowRankCrossAttentionAdapter,
    ):
        super().__init__()
        self.ar_model   = ar_model
        self.rule_model = rule_model
        self.adapter    = adapter

        # Freeze AR transformer
        for p in self.ar_model.parameters():
            p.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, T) long tensor
        Returns final logits (B, T, vocab_size).
        """
        ar_logits, ar_hidden = self.ar_model(idx, return_hidden=True)   # frozen
        rule_logits          = self.rule_model(idx)                      # frozen

        return self.adapter(ar_hidden, ar_logits, rule_logits)

    def trainable_parameters(self):
        return self.adapter.parameters()

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, n_new: int) -> torch.Tensor:
        self.eval()
        tokens  = start_tokens.clone()
        max_len = self.ar_model.max_seq_len
        for _ in range(n_new):
            ctx      = tokens[:, -max_len:]
            logits   = self(ctx)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens   = torch.cat([tokens, next_tok], dim=1)
        return tokens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_patched_model(
    ar_ckpt_path:   str | None = None,
    max_seq_len:    int = 128,
    d_model:        int = 128,
    n_layers:       int = 4,
    n_heads:        int = 4,
    adapter_rank:   int = 16,
    cycle_length:   int = 4,
    force_fallback: bool = False,
    device:         str = "cpu",
) -> PatchedModel:
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
    adapter    = LowRankCrossAttentionAdapter(
                     d_model=d_model, vocab_size=12,
                     rank=adapter_rank, cycle_length=cycle_length).to(device)

    return PatchedModel(ar_model, rule_model, adapter)


if __name__ == "__main__":
    model = build_patched_model(force_fallback=True)
    print(f"Adapter parameters: {sum(p.numel() for p in model.adapter.parameters()):,}")

    x = torch.tensor([[2, 7, 9, 2, 2]], dtype=torch.long)
    out = model(x)
    print("output shape:", out.shape)
    print("predictions :", out.argmax(-1).tolist())
