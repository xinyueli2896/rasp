"""
CP-Transformer Yin-Yang adapter for rule-conditioned bass generation.

Architecture
------------
  BassTracrRuleModel  (models/bass_tracr_rule_model.py)
    Zero-parameter analytical model. Reads starting pitch class from the
    first beat of the sequence (x = pc[0]) and encodes:
      hidden[t] = one-hot((x + OFFSETS[t%4]) % 12) + position encoding
    i.e. the CURRENT pitch class at position t (not next).
    OFFSETS = [0, 5, 7, 0]  — same I-IV-V-I rule as the integer experiment.
    No chord tokens needed; the key is read directly from the generated sequence.

  CPYinyangCrossAttention
    Cross-attention from beat-level AR hidden states to rule hidden states.
    Causal mask: beat t may attend to rule_hidden[0..t].
    At step t the adapter sees hidden[t] = predicted next pitch class → t+1.

  CPYinyangTransformer  (wraps RoFormerSymbolicTransformer)
    Pretrained CP transformer with adapter injected every n_skip global layers.
    The base model is frozen; only yinyang_attn adapters are trained.

    forward(x)
      x : (B, seq_len, subseq_len)  preprocessed CP tokens
      Pitch classes extracted from x[:, :, 1] % 128 % 12 to build rule_hidden.

    global_sampling(x, ...)
      Autoregressive sampling. Starting pitch class read from x[:, 0, 1].
      rule_hidden[t] = embed(key + OFFSETS[t%4]) so the adapter at step t
      naturally attends to rule_hidden[t] (matching PE) for the correct signal.
"""

from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bass_tracr_rule_model import BassTracrRuleModel, TRACR_D_MODEL


# ---------------------------------------------------------------------------
# LoRA wrapper for nn.Linear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for a frozen nn.Linear with a low-rank update.

    output = W_frozen @ x  +  (B @ A @ x) * scale
    Only A and B are trainable. scale = lora_alpha / rank.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear                          # kept frozen by caller
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = (alpha if alpha is not None else float(rank)) / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


# ---------------------------------------------------------------------------
# Cross-attention with bar-aligned causal mask
# ---------------------------------------------------------------------------

class CPYinyangCrossAttention(nn.Module):
    """
    Query  : beat-level AR hidden states  (B, T_beat, d_model)
    Key/Val: beat-level rule hidden states (B, T_beat, rule_d_model)

    Q and K are in the same index space (one entry per beat), so a single
    positional encoding covers both and the causal mask is a plain lower
    triangular: beat t may attend to beats 0..t.
    """

    def __init__(
        self,
        d_model:      int,
        rule_d_model: int,
        embed_dim:    int,
        n_heads:      int   = 8,
        dropout:      float = 0.1,
        max_beats:    int   = 512,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.q_proj   = nn.Linear(d_model,      embed_dim)
        self.k_proj   = nn.Linear(rule_d_model, embed_dim)
        self.v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim,    d_model)
        self.gate      = nn.Parameter(torch.ones(1,))
        self.attn_drop = nn.Dropout(dropout)

        # Single sinusoidal PE shared by Q and K (same beat index space)
        pe  = torch.zeros(max_beats, embed_dim)
        pos = torch.arange(max_beats).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_beats, embed_dim)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        ar_hidden:   torch.Tensor,   # (B, T_q, d_model)
        rule_hidden: torch.Tensor,   # (B, T_k, rule_d_model)
        sub_offset:  int = 0,        # absolute beat index of ar_hidden[:, 0, :]
    ) -> torch.Tensor:
        B, T_q, _ = ar_hidden.shape
        _, T_k, _ = rule_hidden.shape

        Q = self.q_proj(ar_hidden)   + self.pe[:, sub_offset:sub_offset + T_q, :]
        K = self.k_proj(rule_hidden) + self.pe[:, :T_k, :]
        V = self.v_proj(rule_hidden)

        Q = Q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T_q, T_k)

        # Plain causal mask: beat (sub_offset + i) may attend to beat j iff j <= sub_offset + i
        abs_q = torch.arange(sub_offset, sub_offset + T_q, device=ar_hidden.device)
        k_rng = torch.arange(T_k, device=ar_hidden.device)
        causal = abs_q.unsqueeze(1) >= k_rng.unsqueeze(0)              # (T_q, T_k)
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(out) * self.gate


# ---------------------------------------------------------------------------
# Yin-Yang CP transformer
# ---------------------------------------------------------------------------

class CPYinyangTransformer(nn.Module):
    """
    Wraps a pretrained RoFormerSymbolicTransformer and adds rule-conditioned
    Yin-Yang cross-attention adapters.

    Only yinyang_attn adapters are trainable; everything else is frozen.
    The rule model is the analytical BassTracrRuleModel (zero parameters).

    Parameters
    ----------
    base_model   : pretrained RoFormerSymbolicTransformer instance
    adapter_rank : embed_dim for cross-attention projections
    n_skip       : inject adapter every n_skip global transformer layers
    """

    def __init__(
        self,
        base_model,
        adapter_rank:  int  = 256,
        n_skip:        int  = 4,
        lora_rank:     int  = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.base          = base_model
        self.n_skip        = n_skip
        self.lora_rank     = lora_rank
        self.bidirectional = bidirectional

        # Freeze everything in the base model
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Optionally inject LoRA into every attention layer of the base model
        if lora_rank > 0:
            for layer in self.base.model.layer:
                sa = layer.attention.self
                sa.query = LoRALinear(sa.query, lora_rank)
                sa.value = LoRALinear(sa.value, lora_rank)

        n_layers = len(self.base.model.layer)
        assert n_layers % n_skip == 0, \
            f"n_global_layers ({n_layers}) must be divisible by n_skip ({n_skip})"
        n_adapters = n_layers // n_skip

        # Frozen analytical rule model — zero trainable parameters (unused when bidirectional)
        self.rule_model = BassTracrRuleModel()

        self.yinyang_attn = nn.ModuleList([
            CPYinyangCrossAttention(
                d_model      = self.base.hidden_size,
                rule_d_model = TRACR_D_MODEL,
                embed_dim    = adapter_rank,
                n_heads      = 8,
            )
            for _ in range(n_adapters)
        ])

        if bidirectional:
            # Learned AR→rule projection (one per adapter layer).
            # Replaces the TracR rule model: AR hidden states are projected to
            # TRACR_D_MODEL space and used as rule_hidden, so the adapter learns
            # to extract the rule signal from the base model's representations
            # rather than reading pc[:, 0] from the input sequence.
            self.ar_to_rule = nn.ModuleList([
                nn.Linear(self.base.hidden_size, TRACR_D_MODEL)
                for _ in range(n_adapters)
            ])

    # ------------------------------------------------------------------
    # Training forward  (full sequence, layer-by-layer injection)
    # ------------------------------------------------------------------

    def _rule_hidden(self, x_proc: torch.Tensor) -> torch.Tensor:
        """Extract pitch classes from preprocessed tokens and run frozen TracR rule model.

        x_proc : (B, T, subseq_len)  preprocessed CP tokens
          slot 1 = pitch + (dur+1)*128  →  pitch = slot1 % 128  →  pc = pitch % 12
        Rule model reads key = pc[:, 0] and returns hidden (B, T, 16) where
        hidden[t] encodes the CURRENT pitch class at position t: (key + OFFSETS[t%4]) % 12.
        """
        pc = x_proc[:, :, 1] % 128 % 12   # (B, T)
        _, h = self.rule_model(pc, return_hidden=True)   # (B, T, 16)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, subseq_len)  preprocessed CP tokens
        Returns logits of shape (B, seq_len, subseq_len, vocab_size) via local_decode.
        """
        base = self.base
        batch_size, seq_len, _ = x.shape

        rule_hidden = None if self.bidirectional else self._rule_hidden(x)

        h, emb = base.local_encode(x)
        h = h.view(batch_size, seq_len, base.hidden_size)
        sos = base.global_sos.view(1, 1, -1).expand(batch_size, 1, -1)
        h = torch.cat([sos, h[:, :-1]], dim=1)   # (B, seq_len, hidden)

        mask = base.buffered_future_mask(h)

        for i, layer in enumerate(base.model.layer):
            h = layer(h, attention_mask=mask)[0]
            if (i + 1) % self.n_skip == 0:
                adapter_idx = (i + 1) // self.n_skip - 1
                rule_h = self.ar_to_rule[adapter_idx](h) if self.bidirectional else rule_hidden
                h = h + self.yinyang_attn[adapter_idx](h, rule_h, sub_offset=0)

        return base.local_decode(h, emb)

    def loss(self, x: torch.Tensor, pitch_shift: torch.Tensor) -> torch.Tensor:
        x_proc = self.base.preprocess(x, pitch_shift)
        logits = self(x_proc)
        return F.cross_entropy(
            logits.view(-1, self.base.tokenizer.n_tokens),
            x_proc.view(-1),
            ignore_index=self.base.tokenizer.pad_token,
        )

    # ------------------------------------------------------------------
    # Autoregressive sampling  (KV cache for global transformer,
    # adapter applied incrementally at the last position only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def global_sampling(
        self,
        x:           torch.Tensor,   # (B, seed_len, subseq_len)  seed subbeats (preprocessed)
        max_seq_len: int   = 384,
        temperature: float = 1.0,
        sampling_func = None,
    ) -> list:
        base = self.base
        batch_size, seed_len, _ = x.shape

        # Standard mode: pre-compute rule_hidden from dummy constant-key sequence.
        # Bidirectional mode: rule_hidden is computed on-the-fly from AR states each step.
        if not self.bidirectional:
            pc_0   = (x[:, 0, 1] % 128 % 12).clamp(0, 11)           # (B,) starting pitch class
            dummy  = pc_0.unsqueeze(1).expand(-1, max_seq_len)        # (B, max_seq_len)
            _, rule_hidden = self.rule_model(dummy, return_hidden=True)  # (B, max_seq_len, 16)
        else:
            rule_hidden = None

        # Encode seed subbeats
        h_seed, _ = base.local_encode(x)
        h_seed = h_seed.view(batch_size, seed_len, base.hidden_size)
        sos = base.global_sos.view(1, 1, -1).expand(batch_size, 1, -1)
        h = torch.cat([sos, h_seed], dim=1)  # (B, seed_len+1, hidden)

        y = [x[:, i, :] for i in range(seed_len)]

        for i in range(seed_len, max_seq_len):

            # RoFormerEncoder does not accumulate KV cache in this transformers version;
            # run full sequence with causal mask each step (O(n^2) but correct).
            attn_mask = base.buffered_future_mask(h)
            sinusoidal_pos = base.model.embed_positions(h.shape[:-1], 0)[None, None, :, :]

            h_out = h
            for j, layer in enumerate(base.model.layer):
                h_out = layer(h_out, attention_mask=attn_mask, sinusoidal_pos=sinusoidal_pos)[0]

                # Inject adapter after every n_skip layers, modifying last position only
                if (j + 1) % self.n_skip == 0:
                    adapter_idx = (j + 1) // self.n_skip - 1
                    # Bidirectional: project the full AR sequence to rule space on-the-fly;
                    # the cross-attention query (last position) can then attend to the full
                    # AR-derived rule trajectory up to step i.
                    rule_h = self.ar_to_rule[adapter_idx](h_out) if self.bidirectional else rule_hidden
                    correction = self.yinyang_attn[adapter_idx](
                        h_out[:, -1:, :], rule_h, sub_offset=i
                    )
                    h_out = torch.cat([h_out[:, :-1, :], h_out[:, -1:, :] + correction], dim=1)

            # Sample next subbeat from last global hidden state
            y_next = base.local_sampling(
                h_out[:, -1], temperature=temperature,
                global_step=i, sampling_func=sampling_func,
            )
            y.append(y_next)

            # Encode next subbeat and append to h for next step
            h_enc = base.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
            h = torch.cat([h, h_enc], dim=1)

        return y
