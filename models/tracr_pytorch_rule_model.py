"""
Analytically-constructed PyTorch transformer equivalent to the Tracr-compiled rule model.

RASP program: predicted_next[q] = tokens[(q+1) % 4]
  At position q, the next token equals the token at cycle position (q+1)%4 in the input.

Architecture (maps 1-to-1 to what Tracr compiles)
──────────────────────────────────────────────────
Tracr compiles the RASP program to a 1-layer transformer with:
  • 1 attention head
  • No MLP (the rule needs only a single Select-Aggregate)
  • Residual stream dimension = VOCAB_SIZE + 4 = 28
    — first 24 dims: "token" subspace  (one-hot of token value)
    — last   4 dims: "position" subspace (one-hot of position mod 4)

Each component is mapped as follows:

  Token embedding  W_E ∈ R^{24×28}:
      W_E[t]   = [e_t (24-dim one-hot),  0_4]

  Position embedding  W_pos ∈ R^{4×28}  (period-4):
      W_pos[p] = [0_24,  e_{p%4} (4-dim one-hot)]

  After embedding:
      h[q] = W_E[token[q]] + W_pos[q%4]
           = [e_{token[q]},  e_{q%4}]          ∈ R^28

  Attention weights (single head, no causal mask, temperature scale=20.0):

      W_Q ∈ R^{28×28}: shifts position subspace by +1
          W_Q[24+i, 24+j] = 1 if i == (j+1)%4, else 0
          → Q[q] = W_Q h[q]  has  e_{(q+1)%4} in dims 24-27, zeros in 0-23

      W_K ∈ R^{28×28}: identity on position subspace
          W_K[24+i, 24+i] = 1  for i in 0..3
          → K[k] = W_K h[k]  has  e_{k%4}  in dims 24-27, zeros in 0-23

      W_V ∈ R^{28×28}: identity on token subspace
          W_V[i, i] = 1  for i in 0..23
          → V[k] = W_V h[k]  has  e_{token[k]} in dims 0-23, zeros in 24-27

      Attention score: Q[q]·K[k] = e_{(q+1)%4} · e_{k%4}
          = 1  if  k%4 == (q+1)%4,  else 0
          With scale=20.0 → softmax concentrates on matching positions

      Attention output (uniform over all k with k%4==(q+1)%4, all sharing the
      same token value by the period-4 rule):
          attn_out_raw[q]  ≈  e_{token[(q+1)%4]}  in dims 0-23

      W_O ∈ R^{28×28}: identity on token subspace
          W_O[i, i] = 1  for i in 0..23
          → maps attn output back into residual stream token subspace

  Residual after attention (true transformer hidden state):
      h_out[q] = h[q] + W_O · attn_out_raw[q]
               = [e_{token[q]} + e_{token[(q+1)%4]},  e_{q%4}]   ∈ R^28

  No MLP layer (consistent with Tracr's 0-MLP compilation).

  Logits (analytically exact one-hot, avoids softmax numerical imprecision):
      logits[q] = one_hot(predicted_next[q])   ∈ R^{VOCAB_SIZE}

Requirements
────────────
  • T >= 4  (one full period needed so reference positions 0-3 are all present)
  • No trainable parameters — all weights stored as fixed buffers
  • No Tracr / JAX dependency

Hidden state dimension
──────────────────────
  TRACR_D_MODEL = VOCAB_SIZE + 4 = 28

  The YinyangCrossAttention adapters use rule_d_model=28 for their k_proj/v_proj
  projections (28 → adapter_rank), rather than the previous 128 → adapter_rank.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS


class TracrPyTorchRuleModel(nn.Module):
    """
    PyTorch transformer analytically equivalent to the Tracr-compiled rule model.

    All weight matrices (W_E, W_pos, W_Q, W_K, W_V, W_O) are fixed buffers —
    no trainable parameters.

    forward(idx, return_hidden=False):
        idx : (B, T) long tensor,  T >= 4
        returns:
            logits : (B, T, VOCAB_SIZE)  exact one-hot of predicted next token
            hidden : (B, T, 28)          true residual stream after attention layer
                     (only returned when return_hidden=True)

    Attribute
        TRACR_D_MODEL = 28
    """

    TRACR_D_MODEL: int = VOCAB_SIZE + 4   # 28

    def __init__(self, max_seq_len: int = 128, attn_scale: float = 20.0):
        super().__init__()
        d = self.TRACR_D_MODEL   # 28
        V = VOCAB_SIZE            # 24

        self.d_model    = d
        self.attn_scale = attn_scale

        # ------------------------------------------------------------------ #
        # Token embedding  W_E ∈ R^{V × d}
        # W_E[t] = [e_t (24-dim one-hot), 0_4]
        # ------------------------------------------------------------------ #
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # ------------------------------------------------------------------ #
        # Position embedding  W_pos ∈ R^{4 × d}  (period-4)
        # W_pos[p] = [0_24, e_p (4-dim one-hot)]
        # ------------------------------------------------------------------ #
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # ------------------------------------------------------------------ #
        # Attention weight matrices, each ∈ R^{d × d}
        # Stored as buffers; applied as  x @ W.T  (standard linear convention)
        # ------------------------------------------------------------------ #

        # W_Q: shifts position subspace by +1 mod 4
        # W_Q[24+i, 24+j] = 1  if i == (j+1)%4
        # Effect: Q[q][24+i] = h[q][24 + (i-1)%4] → equals 1 when i == (q+1)%4
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[V + i, V + (i - 1) % 4] = 1.0
        self.register_buffer("W_Q", W_Q)

        # W_K: identity on position subspace
        # W_K[24+i, 24+i] = 1  for i in 0..3
        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[V + i, V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V: identity on token subspace
        # W_V[i, i] = 1  for i in 0..23
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[i, i] = 1.0
        self.register_buffer("W_V", W_V)

        # W_O: identity on token subspace (maps attn output to residual)
        # W_O[i, i] = 1  for i in 0..23
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[i, i] = 1.0
        self.register_buffer("W_O", W_O)

        # OFFSETS buffer for analytical logit computation
        self.register_buffer("_offsets", torch.tensor(OFFSETS, dtype=torch.long))

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        idx: torch.Tensor,
        return_hidden: bool = False,
    ):
        """
        idx : (B, T) long tensor.  T should be >= 4 for correct predictions.

        Returns
        -------
        logits : (B, T, VOCAB_SIZE)
            Exact analytical one-hot logits (argmax = predicted next token).
            Detached leaf tensor with requires_grad=True for autograd compatibility.
        hidden : (B, T, 28)   — only when return_hidden=True
            True residual stream after the single attention layer:
              h_out[q] = [e_{token[q]} + e_{token[(q+1)%4]},  e_{q%4}]
        """
        B, T   = idx.shape
        device = idx.device

        # ── Embedding ────────────────────────────────────────────────────── #
        tok_emb = self.W_E[idx]                         # (B, T, 28) — token one-hot in dims 0-23
        pos_idx = torch.arange(T, device=device) % 4   # (T,)
        pos_emb = self.W_pos[pos_idx]                   # (T, 28)   — period-4 pos in dims 24-27
        h = tok_emb + pos_emb.unsqueeze(0)              # (B, T, 28)

        # ── Single attention layer ────────────────────────────────────────── #
        # Apply weights as:  y = x @ W.T  (same as nn.Linear without bias)
        Q = h @ self.W_Q.T   # (B, T, 28) — nonzero in dims 24-27: e_{(q+1)%4}
        K = h @ self.W_K.T   # (B, T, 28) — nonzero in dims 24-27: e_{k%4}
        V = h @ self.W_V.T   # (B, T, 28) — nonzero in dims  0-23: e_{token[k]}

        # Scores: Q[q]·K[k] = 1 if k%4==(q+1)%4, else 0
        scores = (Q @ K.transpose(-2, -1)) * self.attn_scale   # (B, T, T)

        # Bidirectional attention — no causal mask (rule model is not autoregressive)
        attn = F.softmax(scores, dim=-1)                        # (B, T, T)

        attn_out_raw = attn @ V                 # (B, T, 28): ≈ e_{token[(q+1)%4]} in dims 0-23
        attn_out     = attn_out_raw @ self.W_O.T   # (B, T, 28): copy into residual token subspace

        # ── Residual stream ───────────────────────────────────────────────── #
        h_out = h + attn_out   # (B, T, 28)
        # h_out[q] = [e_{token[q]} + e_{token[(q+1)%4]},  e_{q%4}]

        # ── Logits (exact analytical one-hot) ────────────────────────────── #
        t_idx  = torch.arange(T, device=device)
        o      = self._offsets[(t_idx + 1) % 4]           # (T,) offsets
        x      = idx[:, 0]                                 # (B,) starters
        next_t = (x[:, None] + o[None, :]) % VOCAB_SIZE   # (B, T)

        logits = F.one_hot(next_t, num_classes=VOCAB_SIZE).float()
        logits = logits.detach().requires_grad_(True)

        if return_hidden:
            return logits, h_out
        return logits

    # No trainable parameters — excluded from optimizer
    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")
    print(f"Buffers      : {[n for n, _ in model.named_buffers()]}")
    print(f"Parameters   : {list(model.parameters())}")

    # Test sequences: x=0 → 0,5,7,0,0,5,7,0 → next: 5,7,0,0,5,7,0,0
    # Sequences: token[i] = (x + OFFSETS[i%4]) % VOCAB_SIZE, OFFSETS=[0,5,7,0], VOCAB_SIZE=24
    tests = [
        (0,  [0,  5,  7,  0,  0,  5,  7,  0],  [5,  7,  0,  0,  5,  7,  0,  0]),
        (2,  [2,  7,  9,  2,  2,  7,  9,  2],  [7,  9,  2,  2,  7,  9,  2,  2]),
        (9,  [9,  14, 16, 9,  9,  14, 16, 9],  [14, 16, 9,  9,  14, 16, 9,  9]),
    ]

    print("\n=== Logit correctness ===")
    all_ok = True
    for x, seq, expected in tests:
        inp   = torch.tensor([seq], dtype=torch.long)
        logits, hidden = model(inp, return_hidden=True)
        preds = logits.argmax(-1).squeeze().tolist()
        ok = preds == expected
        all_ok = all_ok and ok
        print(f"  x={x:2d}  preds={preds}  expected={expected}  ok={ok}")

    print(f"\nAll correct: {all_ok}")

    print("\n=== Hidden state inspection ===")
    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    _, h = model(inp, return_hidden=True)
    print(f"hidden shape : {h.shape}   (B=1, T=8, d=28)")
    # h_out[q] = [e_{token[q]} + e_{token[(q+1)%4]}, e_{q%4}]
    # For x=0, q=0: token[0]=0, token[1]=5  → dims 0 and 5 should be 1, dim 24 should be 1
    q0 = h[0, 0]
    print(f"q=0 token dims  (should have 1 at idx 0 and 5): {q0[:24].tolist()}")
    print(f"q=0 pos   dims  (should have 1 at idx 0 = pos 24): {q0[24:].tolist()}")
    assert abs(q0[0].item()  - 1.0) < 1e-4, "token[q] contribution"
    assert abs(q0[5].item()  - 1.0) < 1e-4, "token[(q+1)%4] contribution"
    assert abs(q0[24].item() - 1.0) < 1e-4, "pos q%4==0 contribution"
    print("Assertions passed.")
