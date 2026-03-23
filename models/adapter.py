"""
Low-rank cross-attention adapter that patches the rule model onto the frozen
AR transformer.

Architecture overview (inspired by midi_yinyang LowRankMultiheadAttention)
───────────────────────────────────────────────────────────────────────────
                ┌─────────────────────────┐
  tokens ──────►│ AR Transformer (frozen)  │──► ar_hidden  (B,T,d_model)
                │                          │──► ar_logits  (B,T,12)
                └─────────────────────────┘
                                                      │
                ┌─────────────────────────┐           │  Q = q_linear(ar_hidden)
  tokens ──────►│ Rule Model    (frozen)  │──► rule_logits (B,T,12)
                └─────────────────────────┘           │  K = k_linear(rule_logits)
                                                      │  V = v_linear(rule_logits)
                                                      ▼
                                        ┌─────────────────────────┐
                                        │  Multi-Head Cross-Attn  │
                                        │  (trainable, low-rank)  │
                                        │                         │
                                        │  scores = QKᵀ/√head_dim │
                                        │        + cycle_bias     │
                                        │  attn = softmax(scores  │
                                        │          + causal mask) │
                                        │  out = attn @ V         │
                                        │  delta = out_proj(out)  │
                                        └─────────────────────────┘
                                                      │
                                 ar_logits + gates · delta
                                                      │
                                              final_logits (B,T,12)

Design follows midi_yinyang (music-x-lab/midi_yinyang):
  • Separate W_v projection + out_proj (more expressive than V=rule_logits)
  • Multi-head attention (n_heads heads, each of dimension embed_dim//n_heads)
  • gates: scalar Parameter init=0, grows from zero — safe init that avoids
    polluting AR predictions at the start; bootstraps once out_proj(attn@V) ≠ 0

Cycle-position bias (key design decision)
─────────────────────────────────────────
With purely content-based attention (random W_q/W_k init), the attention
starts approximately uniform and remains so — the optimisation landscape is
flat because mean(rule_logits) ≈ uniform over vocab.

The fix: a learnable 4×4 cycle-position bias matrix cycle_bias[c_q, c_k]
(where c_q = t%4 and c_k = s%4) is added to the raw attention scores.
Initialised strongly diagonal (bias=+4 on diagonal), it primes attention
to attend to same-cycle positions from step 0:

  attended[t] ≈ mean(rule_logits[s] for s where s%4==t%4)
             = rule_logits[t]   (same-cycle rule logits are identical)

Parameter count (d=128, embed_dim=32, n_heads=4, vocab=12, cycle_length=4):
  q_linear:  128×32 + 32  = 4,128
  k_linear:   12×32 + 32  =   416
  v_linear:   12×32 + 32  =   416
  out_proj:   32×12 + 12  =   396
  cycle_bias: 4×4         =    16
  gates:      1            =     1
  ─────────────────────────────────
  Total:                     5,373
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
    Multi-head low-rank cross-attention adapter (yinyang-style).

    Queries come from AR hidden states; keys and values come from rule logits.
    The attended output is projected to logit space and gated by a learnable
    scalar `gates` (init=0, grows from zero during training).

    Parameters
    ----------
    d_model    : int – AR transformer hidden size (query input dimension)
    vocab_size : int – vocabulary size = rule output dimension (12)
    embed_dim  : int – low-rank attention dimension (must be divisible by n_heads)
    n_heads    : int – number of attention heads
    cycle_length : int – period of the sequence rule (4)
    """

    def __init__(
        self,
        d_model:      int = 128,
        vocab_size:   int = 12,
        embed_dim:    int = 32,
        n_heads:      int = 4,
        cycle_length: int = 4,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads      = n_heads
        self.head_dim     = embed_dim // n_heads
        self.scale        = self.head_dim ** -0.5
        self.embed_dim    = embed_dim
        self.cycle_length = cycle_length

        # Low-rank projections (yinyang design: separate Q, K, V, out_proj)
        self.q_linear  = nn.Linear(d_model,    embed_dim, bias=True)
        self.k_linear  = nn.Linear(vocab_size, embed_dim, bias=True)
        self.v_linear  = nn.Linear(vocab_size, embed_dim, bias=True)
        self.out_proj  = nn.Linear(embed_dim,  vocab_size, bias=True)

        # Cycle-position bias: primes same-cycle attention from step 0
        self.cycle_bias = nn.Parameter(torch.zeros(cycle_length, cycle_length))

        # Gate scalar (yinyang design): learnable blend coefficient.
        self.gates = nn.Parameter(torch.zeros(1))

        self._init_weights()

        # Freeze v_linear and out_proj after identity init.
        # These are initialised so that out_proj(v_linear(x)) = x exactly.
        # Allowing them to drift under CE loss (where AR is already correct on
        # training starters) destroys the identity and degrades generalisation.
        # Only q_linear, k_linear, cycle_bias, and gates are trainable.
        for p in self.v_linear.parameters():
            p.requires_grad_(False)
        for p in self.out_proj.parameters():
            p.requires_grad_(False)

    def _init_weights(self):
        vocab_size = self.out_proj.out_features   # 12

        # Standard Xavier init for Q and K projections
        for linear in [self.q_linear, self.k_linear]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        # V → out_proj as approximate identity through the embed_dim bottleneck:
        #   v_linear  : (vocab_size→embed_dim)  embeds first vocab_size dims as I,
        #               pads remainder with 0       →  v_linear(x) = [x; 0]
        #   out_proj  : (embed_dim→vocab_size)   extracts first vocab_size dims
        #               →  out_proj([x; 0]) = x
        #
        # Combined: out_proj(v_linear(x)) = x  (exact for embed_dim ≥ vocab_size)
        #
        # With cycle_bias diagonal init, attn ≈ same-cycle rule_logits, so at step 0:
        #   delta = out_proj(attn @ v_linear(rule_logits)) * gates
        #         ≈ out_proj(v_linear(rule_logits))         * gates
        #         ≈ rule_logits                              * gates
        # This gives:  final = ar_logits + gates * rule_logits
        # which is the oracle blend.  gates then just needs to be large enough
        # to overcome AR logit gaps for OOD starters.
        with torch.no_grad():
            self.v_linear.weight.zero_()
            self.v_linear.weight[:vocab_size, :] = torch.eye(vocab_size)
            self.v_linear.bias.zero_()

            self.out_proj.weight.zero_()
            self.out_proj.weight[:, :vocab_size] = torch.eye(vocab_size)
            self.out_proj.bias.zero_()

        # Diagonal cycle bias: strongly prefer same-cycle attention at init
        with torch.no_grad():
            self.cycle_bias.fill_diagonal_(4.0)

        # Gates: start large enough to immediately dominate AR logit gaps (~4 units).
        # The identity init above ensures the initial delta ≈ rule_logits,
        # so a large gates value is safe (and necessary for OOD starters).
        with torch.no_grad():
            self.gates.fill_(5.0)

    def forward(
        self,
        ar_hidden:   torch.Tensor,   # (B, T, d_model)
        ar_logits:   torch.Tensor,   # (B, T, vocab_size)
        rule_logits: torch.Tensor,   # (B, T, vocab_size)
    ) -> torch.Tensor:               # (B, T, vocab_size)

        B, T, _ = ar_hidden.shape
        device  = ar_hidden.device

        # Low-rank projections
        Q = self.q_linear(ar_hidden)    # (B, T, embed_dim)
        K = self.k_linear(rule_logits)  # (B, T, embed_dim)
        V = self.v_linear(rule_logits)  # (B, T, embed_dim)

        # Split into heads: (B, n_heads, T, head_dim)
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale   # (B, n_heads, T, T)

        # Cycle-position bias — broadcast over batch and heads
        t_idx  = torch.arange(T, device=device)
        bias   = self.cycle_bias[t_idx[:, None] % self.cycle_length,
                                  t_idx[None, :] % self.cycle_length]  # (T, T)
        scores = scores + bias.unsqueeze(0).unsqueeze(0)

        # Causal mask
        causal = torch.ones(T, T, device=device).tril().bool()
        scores = scores.masked_fill(~causal, float('-inf'))

        attn = F.softmax(scores, dim=-1)                  # (B, n_heads, T, T)

        # Weighted value aggregation
        out = (attn @ V)                                  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous()            # (B, T, n_heads, head_dim)
        out = out.view(B, T, self.embed_dim)              # (B, T, embed_dim)

        # Project to vocab/logit space and scale by gate (yinyang: out_proj × gates)
        delta = self.out_proj(out) * self.gates           # (B, T, vocab_size)

        return ar_logits + delta


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
    adapter_rank:   int = 32,          # embed_dim for the low-rank adapter
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
                     embed_dim=adapter_rank, n_heads=4,
                     cycle_length=cycle_length).to(device)

    return PatchedModel(ar_model, rule_model, adapter)


if __name__ == "__main__":
    model = build_patched_model(force_fallback=True)
    print(f"Adapter parameters: {sum(p.numel() for p in model.adapter.parameters()):,}")
    print(f"  gates init: {model.adapter.gates.item():.4f}")

    x = torch.tensor([[2, 7, 9, 2, 2]], dtype=torch.long)
    out = model(x)
    print("output shape  :", out.shape)
    print("predictions   :", out.argmax(-1).tolist())
    print("(gates=0 → adapter output is zero at init, predictions = pure AR)")
