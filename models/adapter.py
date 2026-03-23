"""
Cross-attention adapter that patches the rule model onto the frozen AR transformer.

Architecture overview
─────────────────────
                ┌─────────────────────────┐
  tokens ──────►│ AR Transformer (frozen)  │──► ar_hidden  (B,T,d_model)
                │                          │──► ar_logits  (B,T,12)
                └─────────────────────────┘
                                                      │  Q = q_linear(ar_hidden)
                ┌─────────────────────────┐           │
  tokens ──────►│ Rule Model    (frozen)  │──► rule_logits (B,T,12)
                └─────────────────────────┘           │  K = k_linear(rule_logits)
                                                      │
                         correction = rule_logits - ar_logits
                                                      │  V = v_linear(correction)
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
                                        │  h = out_proj(out)      │  ← hidden space
                                        └─────────────────────────┘
                                                      │  gates · h
                                                      ▼
                                         ar_hidden + gates · h    ← residual in hidden space
                                                      │
                                              lm_head (frozen)
                                                      │
                                              final_logits (B,T,12)

Correction signal
─────────────────
Values are computed from the *correction* needed: correction = rule_logits - ar_logits.
This is the direction in logit space the AR model needs to move.

For the residual to satisfy  lm_head(ar_hidden + h) ≈ rule_logits  we need:
  W_E h ≈ correction        where W_E = lm_head.weight  (12 × d_model)
  W_E (out_proj(attn @ v_linear(correction))) ≈ correction

With the diagonal cycle_bias, attn[t] ≈ same-cycle positions, so:
  out[t] ≈ v_linear(correction[t])
  h[t]   ≈ out_proj(v_linear(correction[t])) = (out_proj.weight @ v_linear.weight) @ correction[t]

We want  W_E @ (out_proj.weight @ v_linear.weight) ≈ I_vocab.
Setting  out_proj.weight @ v_linear.weight = W_E⁺  (pseudoinverse of W_E) achieves this:
  W_E @ W_E⁺ = I_vocab  (exact, since W_E is full row-rank for any non-degenerate embedding)

Pseudoinverse init (init_from_lmhead)
──────────────────────────────────────
  W_E⁺ = W_E.T @ (W_E @ W_E.T)⁻¹   shape (d_model, vocab_size)
  SVD factorisation through embed_dim bottleneck:
    W_E⁺ ≈ U @ diag(S) @ Vh  (top-r singular vectors, r = min(embed_dim, vocab_size))
    out_proj.weight  ← U[:, :r] * sqrt(S[:r])    (d_model × embed_dim, fills first r cols)
    v_linear.weight  ← sqrt(S[:r]) * Vh[:r, :]   (embed_dim × vocab_size, fills first r rows)
  With gates=1: lm_head(ar_hidden + h) ≈ ar_logits + correction = rule_logits  ✓

This gives the same "oracle blend" behaviour as the logit-space identity init, but
implemented entirely in hidden-state space.

Cycle-position bias
───────────────────
A learnable 4×4 bias matrix cycle_bias[c_q, c_k] (c = t % 4) is added to
the raw attention scores.  Diagonal init (+4) primes same-cycle attention
from step 0, preventing the flat-landscape problem with uniform attention.

Parameter count (d=128, embed_dim=32, n_heads=4, vocab=12, cycle_length=4):
  q_linear:  128×32 + 32  = 4,128
  k_linear:   12×32 + 32  =   416
  v_linear:   12×32 + 32  =   416
  out_proj:   32×128 + 128 = 4,224   ← embed_dim → d_model (hidden space)
  cycle_bias: 4×4          =    16
  gates:      1             =     1
  ─────────────────────────────────
  Total:                     9,201
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
    Multi-head low-rank cross-attention adapter operating in hidden space.

    Queries come from AR hidden states; keys from rule logits; values from
    the correction signal (rule_logits - ar_logits).  The attended output is
    projected to d_model space and added as a residual to ar_hidden.

    Call init_from_lmhead(lm_head_weight) after construction to set up the
    pseudoinverse initialisation that gives oracle-blend behaviour at step 0.

    Parameters
    ----------
    d_model      : int – AR transformer hidden size
    vocab_size   : int – vocabulary size (12)
    embed_dim    : int – low-rank attention dimension (divisible by n_heads)
    n_heads      : int – number of attention heads
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

        self.q_linear  = nn.Linear(d_model,    embed_dim, bias=True)
        self.k_linear  = nn.Linear(vocab_size, embed_dim, bias=True)
        self.v_linear  = nn.Linear(vocab_size, embed_dim, bias=True)  # input: rule_logits
        self.out_proj  = nn.Linear(embed_dim,  d_model,   bias=True)  # output: hidden space
        # Direct AR correction: frozen linear that maps ar_logits → hidden space
        # Initialized to -W_E^+ so it cancels the ar_logits component.
        self.ar_proj   = nn.Linear(vocab_size, d_model,   bias=False)

        self.cycle_bias = nn.Parameter(torch.zeros(cycle_length, cycle_length))
        self.gates      = nn.Parameter(torch.zeros(1))

        self._init_weights_default()

    def _init_weights_default(self):
        """Fallback init used before init_from_lmhead is called."""
        for linear in [self.q_linear, self.k_linear, self.v_linear]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.ar_proj.weight)
        with torch.no_grad():
            self.gates.fill_(1.0)
            self.cycle_bias.fill_diagonal_(4.0)

    @torch.no_grad()
    def init_from_lmhead(self, W_E: torch.Tensor):
        """
        Set v_linear, out_proj, and ar_proj for oracle behaviour at step 0.

        W_E : (vocab_size, d_model) – lm_head.weight from the frozen AR model.

        Two-path oracle derivation:
          h = out_proj(attn @ v_linear(rule_logits)) + ar_proj(ar_logits)

          We want  lm_head(ar_hidden + gates·h) = rule_logits
          ⟹ ar_logits + W_E·h = rule_logits
          ⟹ W_E·(out_proj·v_linear·rule_logits + ar_proj·ar_logits) = rule_logits - ar_logits

          Setting:
            out_proj.weight @ v_linear.weight  =  W_E⁺     →  W_E·W_E⁺·rule = rule
            ar_proj.weight                     = -W_E⁺     →  W_E·(-W_E⁺)·ar = -ar
          Combined: ar + rule - ar = rule  ✓

          V = v_linear(rule_logits) is CONSTANT per cycle position, so causal
          aggregation over past same-cycle positions is exact (no averaging error).
        """
        vocab_size, d_model = W_E.shape
        r = min(self.embed_dim, vocab_size)

        # Pseudoinverse  W_E⁺ = W_E.T @ (W_E @ W_E.T)⁻¹   shape (d_model, vocab_size)
        W_E_pinv = W_E.T @ torch.linalg.inv(W_E @ W_E.T)

        # Factorize W_E⁺ through embed_dim bottleneck: out_proj.weight @ v_linear.weight = W_E⁺
        U, S, Vh = torch.linalg.svd(W_E_pinv, full_matrices=False)
        S_sqrt = S[:r].sqrt()

        self.out_proj.weight.zero_()
        self.out_proj.weight[:, :r] = U[:, :r] * S_sqrt[None, :]   # (d_model, embed_dim)

        self.v_linear.weight.zero_()
        self.v_linear.weight[:r, :] = S_sqrt[:, None] * Vh[:r, :]  # (embed_dim, vocab_size)

        self.out_proj.bias.zero_()
        self.v_linear.bias.zero_()

        # ar_proj: direct path that cancels ar_logits contribution
        self.ar_proj.weight.copy_(-W_E_pinv)   # (d_model, vocab_size) = -W_E⁺

        # gates=1: full oracle correction from step 0
        self.gates.fill_(1.0)

    def forward(
        self,
        ar_hidden:   torch.Tensor,   # (B, T, d_model)
        ar_logits:   torch.Tensor,   # (B, T, vocab_size)
        rule_logits: torch.Tensor,   # (B, T, vocab_size)
    ) -> torch.Tensor:               # (B, T, d_model) modified hidden state

        B, T, _ = ar_hidden.shape
        device  = ar_hidden.device

        Q = self.q_linear(ar_hidden)                  # (B, T, embed_dim)
        K = self.k_linear(rule_logits)                # (B, T, embed_dim)
        V = self.v_linear(rule_logits)                # (B, T, embed_dim)  — constant per cycle

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scores = (Q @ K.transpose(-2, -1)) * self.scale   # (B, n_heads, T, T)
        t_idx  = torch.arange(T, device=device)
        bias   = self.cycle_bias[t_idx[:, None] % self.cycle_length,
                                  t_idx[None, :] % self.cycle_length]
        scores = scores + bias.unsqueeze(0).unsqueeze(0)

        causal = torch.ones(T, T, device=device).tril().bool()
        scores = scores.masked_fill(~causal, float('-inf'))

        attn = F.softmax(scores, dim=-1)              # (B, n_heads, T, T)

        out = (attn @ V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.embed_dim)          # (B, T, embed_dim)

        h_rule = self.out_proj(out)                    # (B, T, d_model)  ≈ W_E⁺ @ rule_logits
        h_ar   = self.ar_proj(ar_logits)               # (B, T, d_model)  = -W_E⁺ @ ar_logits
        return ar_hidden + self.gates * (h_rule + h_ar)


# ---------------------------------------------------------------------------
# Full patched model: AR + Rule + Adapter
# ---------------------------------------------------------------------------

class PatchedModel(nn.Module):
    """
    Combines the frozen AR transformer, the frozen rule model, and the
    trainable cross-attention adapter.

    Forward pass:
      1. ar_hidden, ar_logits = AR transformer            (frozen)
      2. rule_logits           = rule model               (frozen)
      3. modified_hidden = adapter(ar_hidden, ar_logits, rule_logits)  (trainable)
      4. final_logits    = lm_head(modified_hidden)       (frozen)
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

        for p in self.ar_model.parameters():
            p.requires_grad_(False)

        # Oracle init: set v_linear/out_proj using the frozen lm_head weights.
        # Zero out q_linear and k_linear so Q·K = 0 for all (query, key) pairs;
        # attention is then determined solely by cycle_bias (generalises perfectly).
        # Freeze all four projections — only cycle_bias (16) and gates (1) train.
        self.adapter.init_from_lmhead(self.ar_model.lm_head.weight.detach())
        with torch.no_grad():
            self.adapter.q_linear.weight.zero_()
            self.adapter.q_linear.bias.zero_()
            self.adapter.k_linear.weight.zero_()
            self.adapter.k_linear.bias.zero_()
        for name in ('v_linear', 'out_proj', 'q_linear', 'k_linear', 'ar_proj'):
            for p in getattr(self.adapter, name).parameters():
                p.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        ar_logits, ar_hidden = self.ar_model(idx, return_hidden=True)   # frozen
        rule_logits          = self.rule_model(idx)                      # frozen

        modified_hidden = self.adapter(ar_hidden, ar_logits, rule_logits)
        return self.ar_model.lm_head(modified_hidden)                   # frozen

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
    adapter_rank:   int = 32,
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

    # With oracle init, predictions at step 0 should follow the rule
    x = torch.tensor([[2, 7, 9, 2, 2]], dtype=torch.long)
    out = model(x)
    print("output shape  :", out.shape)
    print("predictions   :", out.argmax(-1).tolist())
    # Expected: [7, 9, 2, 2, 7] (next token by rule for starter x=2)
    print("expected      : [[7, 9, 2, 2, 7]]")
