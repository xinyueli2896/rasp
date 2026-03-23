"""
Cross-attention adapter that patches the rule model onto the frozen AR transformer.

Architecture overview
─────────────────────
                ┌─────────────────────────┐
  tokens ──────►│ AR Transformer (frozen)  │──► ar_hidden  (B,T,d_model)
                │                          │──► ar_logits  (B,T,vocab_size)
                └─────────────────────────┘
                                                      │  Q = q_linear(ar_hidden)
                ┌─────────────────────────┐           │
  tokens ──────►│ Rule model (frozen)     │──► rule_hidden (B,T,rule_d_model)
                │  rule_tok_emb[next_tok] │           │  K = k_linear(rule_hidden)
                └─────────────────────────┘           │  V = v_linear(rule_hidden)
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
                                        │  h_rule = out_proj(out) │  ← AR hidden space
                                        └─────────────────────────┘
                                                      │
                                           + ar_proj(ar_logits)   ← direct AR correction
                                                      │  gates · h
                                                      ▼
                                         ar_hidden + gates · h    ← residual in hidden space
                                                      │
                                              lm_head (frozen)
                                                      │
                                              final_logits (B,T,vocab_size)

rule_hidden: rule model's private hidden state
──────────────────────────────────────────────
rule_hidden is taken directly from the rule model's own embedding table:

    rule_hidden[b, t] = rule_tok_emb[predicted_next[b, t]]

rule_tok_emb lives in rule_d_model-dimensional space, which is independent
of the AR model's d_model and vocabulary.  k_linear and v_linear project from
rule_d_model into the adapter's embed_dim.

Oracle SVD initialisation (init_from_lmhead)
─────────────────────────────────────────────
W_E : (vocab_size, d_model)     – AR lm_head.weight
R_E : (vocab_size, rule_d_model) – rule model's rule_tok_emb

Goal: W_E @ out_proj.weight @ v_linear.weight @ R_E.T = I_vocab
i.e.  W_E @ M @ R_E.T = I   where M = out_proj.weight @ v_linear.weight

Solution:  M = W_E⁺ @ (R_E.T)⁺  (pseudoinverses)
  Proof: W_E @ M @ R_E.T = (W_E @ W_E⁺) @ ((R_E.T)⁺ @ R_E.T)
                          = I @ I = I  ✓   (holds when both have full row rank)

M has shape (d_model × rule_d_model).  Factorised through bottleneck via SVD of M:
  M = U_M S_M Vh_M
  out_proj.weight[:, :r] = U_M[:, :r] * sqrt(S_M[:r])   (d_model  × embed_dim)
  v_linear.weight[:r, :] = sqrt(S_M[:r]) * Vh_M[:r, :]  (embed_dim × rule_d_model)

ar_proj: direct path for −ar_logits
  ar_proj.weight = −W_E⁺ = −W_E.T @ (W_E @ W_E.T)⁻¹

Note: when R_E = W_E (shared vocab, old behaviour), M = W_E⁺ @ (W_E.T)⁺ = Vh.T S⁻² Vh,
which exactly matches the previous SVD init.

Cycle-position bias
───────────────────
A learnable 4×4 bias matrix cycle_bias[c_q, c_k] (c = t % 4) is added to
the raw attention scores.  Diagonal init (+4) primes same-cycle attention
from step 0, preventing the flat-landscape problem with uniform attention.

Parameter count (d_model=128, rule_d_model=128, embed_dim=32, n_heads=4, vocab=12, cycle=4):
  q_linear:  128×32 + 32  = 4,128
  k_linear:  128×32 + 32  = 4,128   ← rule_d_model input
  v_linear:  128×32 + 32  = 4,128   ← rule_d_model input
  out_proj:   32×128 + 128 = 4,224
  ar_proj:    12×128       = 1,536   (no bias, frozen)
  cycle_bias: 4×4          =    16
  gates:      1             =     1
  ─────────────────────────────────
  Total trainable (cycle_bias + gates): 17
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

    Queries come from AR hidden states; keys and values come from rule_hidden,
    the rule model's private hidden state (rule_tok_emb[predicted_next_token]).
    rule_hidden lives in rule_d_model space, which is independent of d_model.
    The attended output is projected to d_model space and added as a residual
    to ar_hidden.

    Call init_from_lmhead(W_E, R_E) after construction to set up the
    SVD oracle initialisation that gives exact rule predictions at step 0.

    Parameters
    ----------
    d_model      : int – AR transformer hidden size
    rule_d_model : int – rule model hidden size (rule_tok_emb dimension)
    vocab_size   : int – AR vocabulary size (12)
    embed_dim    : int – low-rank attention dimension (divisible by n_heads)
    n_heads      : int – number of attention heads
    cycle_length : int – period of the sequence rule (4)
    """

    def __init__(
        self,
        d_model:      int = 128,
        rule_d_model: int = 128,
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

        self.q_linear  = nn.Linear(d_model,      embed_dim, bias=True)
        self.k_linear  = nn.Linear(rule_d_model, embed_dim, bias=True)  # input: rule_hidden
        self.v_linear  = nn.Linear(rule_d_model, embed_dim, bias=True)  # input: rule_hidden
        self.out_proj  = nn.Linear(embed_dim,    d_model,   bias=True)  # output: AR hidden space
        # Direct AR correction: frozen linear that maps ar_logits → hidden space
        # Initialized to -W_E^+ so it cancels the ar_logits component.
        self.ar_proj   = nn.Linear(vocab_size, d_model,  bias=False)

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
    def init_from_lmhead(self, W_E: torch.Tensor, R_E: torch.Tensor):
        """
        Set v_linear, out_proj, and ar_proj for oracle behaviour at step 0.

        W_E : (vocab_size, d_model)      – AR lm_head.weight
        R_E : (vocab_size, rule_d_model) – rule model's rule_tok_emb

        Goal: W_E @ out_proj.weight @ v_linear.weight @ R_E.T = I_vocab
              i.e. W_E @ M @ R_E.T = I,  M = out_proj.weight @ v_linear.weight

        Solution: M = W_E⁺ @ (R_E.T)⁺  (pseudoinverses)
          Proof: W_E @ M @ R_E.T = (W_E @ W_E⁺) @ ((R_E.T)⁺ @ R_E.T) = I ✓
          (holds when W_E and R_E both have full row rank, true for vocab_size ≤ d_model
           and vocab_size ≤ rule_d_model with random R_E a.s.)

        M is factorised through the embed_dim bottleneck via SVD of M:
          M = U_M S_M Vh_M
          out_proj.weight[:, :r] = U_M[:, :r] * sqrt(S_M[:r])
          v_linear.weight[:r, :] = sqrt(S_M[:r]) * Vh_M[:r, :]

        ar_proj.weight = -W_E⁺ cancels the ar_logits component.
        """
        r = min(self.embed_dim, W_E.shape[0])

        W_E_pinv = torch.linalg.pinv(W_E)    # (d_model,      vocab_size)
        R_E_T_pinv = torch.linalg.pinv(R_E.T)  # (vocab_size, rule_d_model)
        M = W_E_pinv @ R_E_T_pinv            # (d_model, rule_d_model)

        # Rank-r factorization of M via SVD
        U_M, S_M, Vh_M = torch.linalg.svd(M, full_matrices=False)
        S_sqrt = S_M[:r].sqrt()

        self.out_proj.weight.zero_()
        self.out_proj.weight[:, :r] = U_M[:, :r] * S_sqrt[None, :]    # (d_model, embed_dim)

        self.v_linear.weight.zero_()
        self.v_linear.weight[:r, :] = S_sqrt[:, None] * Vh_M[:r, :]   # (embed_dim, rule_d_model)

        self.v_linear.bias.zero_()
        self.out_proj.bias.zero_()

        # ar_proj: -W_E⁺  shape (d_model, vocab_size)
        self.ar_proj.weight.copy_(-W_E_pinv)

        # gates=1: full oracle correction from step 0
        self.gates.fill_(1.0)

    def forward(
        self,
        ar_hidden:   torch.Tensor,   # (B, T, d_model)
        ar_logits:   torch.Tensor,   # (B, T, vocab_size)
        rule_hidden: torch.Tensor,   # (B, T, d_model)  — RASP hidden state
    ) -> torch.Tensor:               # (B, T, d_model) modified hidden state

        B, T, _ = ar_hidden.shape
        device  = ar_hidden.device

        Q = self.q_linear(ar_hidden)                  # (B, T, embed_dim)
        K = self.k_linear(rule_hidden)                # (B, T, embed_dim)
        V = self.v_linear(rule_hidden)                # (B, T, embed_dim)  — constant per cycle

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
      1. ar_logits, ar_hidden = AR transformer                  (frozen)
      2. rule_logits, rule_hidden = rule model                  (frozen)
         rule_hidden comes from rule_model.rule_tok_emb — the rule model's
         own private embedding space, independent of the AR vocabulary.
      3. modified_hidden = adapter(ar_hidden, ar_logits, rule_hidden)  (trainable)
      4. final_logits    = lm_head(modified_hidden)              (frozen)
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

        # Oracle init: set v_linear/out_proj using AR lm_head and rule embedding.
        # Zero out q_linear and k_linear so Q·K = 0 for all (query, key) pairs;
        # attention is then determined solely by cycle_bias (generalises perfectly).
        # Freeze all four projections — only cycle_bias (16) and gates (1) train.
        self.adapter.init_from_lmhead(
            self.ar_model.lm_head.weight.detach(),
            self.rule_model.rule_tok_emb.detach(),
        )
        with torch.no_grad():
            self.adapter.q_linear.weight.zero_()
            self.adapter.q_linear.bias.zero_()
            self.adapter.k_linear.weight.zero_()
            self.adapter.k_linear.bias.zero_()
        for name in ('v_linear', 'out_proj', 'q_linear', 'k_linear', 'ar_proj'):
            for p in getattr(self.adapter, name).parameters():
                p.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        ar_logits, ar_hidden         = self.ar_model(idx, return_hidden=True)   # frozen
        rule_logits, rule_hidden     = self.rule_model(idx, return_hidden=True) # frozen

        # rule_hidden (B, T, rule_d_model): rule model's private embedding of its
        # predicted next token — independent of the AR model's vocabulary and lm_head.
        modified_hidden = self.adapter(ar_hidden, ar_logits, rule_hidden)
        return self.ar_model.lm_head(modified_hidden)                           # frozen

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
    rule_d_model:   int = 128,
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
                                  rule_d_model=rule_d_model,
                                  force_fallback=force_fallback).to(device)
    adapter    = LowRankCrossAttentionAdapter(
                     d_model=d_model, rule_d_model=rule_d_model, vocab_size=12,
                     embed_dim=adapter_rank, n_heads=4,
                     cycle_length=cycle_length).to(device)

    return PatchedModel(ar_model, rule_model, adapter)


if __name__ == "__main__":
    model = build_patched_model(force_fallback=True)
    print(f"Adapter parameters: {sum(p.numel() for p in model.adapter.parameters()):,}")
    print(f"  trainable params: {sum(p.numel() for p in model.adapter.parameters() if p.requires_grad):,}")
    print(f"  gates init: {model.adapter.gates.item():.4f}")

    # With oracle init, predictions at step 0 should follow the rule.
    # Sequence x=2: tokens = [2, 7, 9, 2, 2, 7, 9, 2, ...]
    # Expected next-token: [7, 9, 2, 2, 7, 9, 2, 2]
    x = torch.tensor([[2, 7, 9, 2, 2, 7, 9, 2]], dtype=torch.long)
    out = model(x)
    print("output shape  :", out.shape)
    print("predictions   :", out.argmax(-1).tolist())
    print("expected      : [[7, 9, 2, 2, 7, 9, 2, 2]]")

    # Verify rule hidden extraction: rule_hidden == rule_tok_emb[predicted_next]
    rule_logits, rule_hidden = model.rule_model(x, return_hidden=True)
    # predicted next tokens for x=2, T=8: (2 + OFFSETS[(t+1)%4]) % 12
    next_t = rule_logits[0].argmax(-1)  # ground-truth predictions
    rule_hidden_ref = model.rule_model.rule_tok_emb[next_t]
    print("rule_hidden shape:", rule_hidden.shape)
    print("hidden matches rule_tok_emb lookup:",
          torch.allclose(rule_hidden[0], rule_hidden_ref, atol=1e-6))
