"""
CP-Transformer Yin-Yang adapter for chord-conditioned bass generation.

Architecture
------------
  ChordRuleModel
    Small bidirectional transformer over bar-level chord tokens.
    Input : (B, n_bars)                  int64
    Output: (B, n_bars, rule_d_model)    float

  CPYinyangCrossAttention
    Cross-attention from subbeat AR hidden states to bar-level rule hidden.
    Key difference from the RASP version: the causal mask is bar-aligned —
    subbeat t can attend to bar b iff t // subbeats_per_bar >= b.
    This is correct when T_q (subbeats) >> T_k (bars).

  CPYinyangTransformer  (extends RoFormerSymbolicTransformer)
    Pretrained CP transformer with adapter injected at the global level.
    The base model is frozen; only ChordRuleModel + adapters are trained.

    forward(x, chord_tokens)
      Layer-by-layer global transformer; adapter injected every n_skip layers.

    global_sampling_chord(x, chord_tokens, ...)
      Autoregressive sampling with chord conditioning.
      Uses KV cache for the global transformer; adapter applied at each step
      to the most recent position only (valid because of causal mask).

Dataset note
------------
  For pretraining the base CP transformer use bass-only tracks from the LA
  dataset (GM programs 32-39). The existing FramedDataset + preprocess_midi
  pipeline works unchanged; just filter by program in preprocess_midi.

  For adapter fine-tuning, pair bass tracks with XF chord annotations from
  the RWC dataset (xf_midi.chords). Use chord_tokenizer.chords_to_bar_tokens
  to convert per-song chord events into bar-token sequences that align with
  the FramedDataset windows.
"""

from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from midi_adapter.chord_tokenizer import N_CHORD_TOKENS, NO_CHORD_TOKEN, N_QUALITIES
from models.bass_tracr_rule_model import BassTracrRuleModel, TRACR_D_MODEL

SUBBEATS_PER_BAR = 16   # 4/4, beat_div=4


# ---------------------------------------------------------------------------
# Cross-attention with bar-aligned causal mask
# ---------------------------------------------------------------------------

class CPYinyangCrossAttention(nn.Module):
    """
    Query  : subbeat AR hidden states    (B, T_sub, d_model)
    Key/Val: bar-level rule hidden states (B, n_bars, rule_d_model)
    Causal : subbeat t attends to bar b  iff  t // subbeats_per_bar >= b
    """

    def __init__(
        self,
        d_model:          int,
        rule_d_model:     int,
        embed_dim:        int,
        n_heads:          int   = 8,
        dropout:          float = 0.1,
        max_subbeats:     int   = 512,
        max_bars:         int   = 64,
        subbeats_per_bar: int   = SUBBEATS_PER_BAR,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads          = n_heads
        self.head_dim         = embed_dim // n_heads
        self.embed_dim        = embed_dim
        self.subbeats_per_bar = subbeats_per_bar

        self.q_proj   = nn.Linear(d_model,      embed_dim)
        self.k_proj   = nn.Linear(rule_d_model, embed_dim)
        self.v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim,    d_model)
        self.gate      = nn.Parameter(torch.ones(1))
        self.attn_drop = nn.Dropout(dropout)

        # Separate positional encodings for subbeat (Q) and bar (K) spaces
        # so that Q[t] and K[b] have compatible scales without conflating indices.
        def _make_pe(length, dim):
            pe  = torch.zeros(length, dim)
            pos = torch.arange(length).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe.unsqueeze(0)   # (1, length, dim)

        self.register_buffer('pe_sub', _make_pe(max_subbeats, embed_dim))  # subbeat positions
        self.register_buffer('pe_bar', _make_pe(max_bars,     embed_dim))  # bar positions

        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        ar_hidden:   torch.Tensor,          # (B, T_sub, d_model)
        rule_hidden: torch.Tensor,          # (B, n_bars, rule_d_model)
        sub_offset:  int = 0,               # first subbeat index (for incremental sampling)
    ) -> torch.Tensor:
        B, T_sub, _ = ar_hidden.shape
        _, n_bars, _ = rule_hidden.shape

        Q = self.q_proj(ar_hidden) + self.pe_sub[:, sub_offset:sub_offset + T_sub, :]
        K = self.k_proj(rule_hidden) + self.pe_bar[:, :n_bars, :]
        V = self.v_proj(rule_hidden)

        Q = Q.view(B, T_sub,  self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, n_bars, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, n_bars, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T_sub, n_bars)

        # Bar-aligned causal mask: subbeat t (absolute index sub_offset+i)
        # may attend to bar b iff (sub_offset + i) // subbeats_per_bar >= b
        abs_sub = torch.arange(sub_offset, sub_offset + T_sub, device=ar_hidden.device)
        bar_of  = abs_sub // self.subbeats_per_bar                             # (T_sub,)
        bar_rng = torch.arange(n_bars, device=ar_hidden.device)                # (n_bars,)
        causal  = bar_of.unsqueeze(1) >= bar_rng.unsqueeze(0)                  # (T_sub, n_bars)
        scores  = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T_sub, self.embed_dim)
        return self.out_proj(out) * self.gate


# ---------------------------------------------------------------------------
# Chord rule model
# ---------------------------------------------------------------------------

class ChordRuleModel(nn.Module):
    """
    Bidirectional transformer over bar-level chord tokens.
    Bidirectional is appropriate because the full chord chart is known at
    generation time (chords are the conditioning input, not generated).
    """

    def __init__(
        self,
        rule_d_model: int = 128,
        n_layers:     int = 2,
        n_heads:      int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(N_CHORD_TOKENS, rule_d_model,
                                  padding_idx=NO_CHORD_TOKEN)
        layer = nn.TransformerEncoderLayer(
            d_model=rule_d_model, nhead=n_heads,
            dim_feedforward=rule_d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                      enable_nested_tensor=False)

    def forward(self, chord_tokens: torch.Tensor) -> torch.Tensor:
        # chord_tokens: (B, n_bars)
        key_pad_mask = (chord_tokens == NO_CHORD_TOKEN)   # True = ignore
        x = self.embed(chord_tokens)                      # (B, n_bars, rule_d_model)
        return self.transformer(x, src_key_padding_mask=key_pad_mask)


# ---------------------------------------------------------------------------
# Yin-Yang CP transformer
# ---------------------------------------------------------------------------

class CPYinyangTransformer(nn.Module):
    """
    Wraps a pretrained RoFormerSymbolicTransformer and adds chord-conditioned
    Yin-Yang cross-attention adapters.

    Only ChordRuleModel and yinyang_attn are trainable; everything else is frozen.

    Parameters
    ----------
    base_model       : pretrained RoFormerSymbolicTransformer instance
    rule_d_model     : hidden dim of the chord rule model
    adapter_rank     : embed_dim for cross-attention projections
    n_skip           : inject adapter every n_skip global transformer layers
    subbeats_per_bar : 16 for 4/4, beat_div=4
    """

    def __init__(
        self,
        base_model,
        adapter_rank:     int = 256,
        n_skip:           int = 4,
        subbeats_per_bar: int = SUBBEATS_PER_BAR,
    ):
        super().__init__()
        self.base             = base_model
        self.subbeats_per_bar = subbeats_per_bar
        self.n_skip           = n_skip

        # Freeze everything in the base model
        for p in self.base.parameters():
            p.requires_grad_(False)

        n_layers = len(self.base.model.layer)
        assert n_layers % n_skip == 0, \
            f"n_global_layers ({n_layers}) must be divisible by n_skip ({n_skip})"
        n_adapters = n_layers // n_skip

        # Frozen analytical rule model — no trainable parameters
        self.rule_model = BassTracrRuleModel()

        self.yinyang_attn = nn.ModuleList([
            CPYinyangCrossAttention(
                d_model          = self.base.hidden_size,
                rule_d_model     = TRACR_D_MODEL,
                embed_dim        = adapter_rank,
                n_heads          = 8,
                subbeats_per_bar = subbeats_per_bar,
            )
            for _ in range(n_adapters)
        ])

    # ------------------------------------------------------------------
    # Training forward  (full sequence, layer-by-layer injection)
    # ------------------------------------------------------------------

    def _rule_hidden(self, chord_tokens: torch.Tensor) -> torch.Tensor:
        """Extract roots from chord tokens and run frozen TracR rule model."""
        roots = chord_tokens // N_QUALITIES          # (B, n_bars) in 0-11
        roots = roots.clamp(0, 11)
        _, h  = self.rule_model(roots, return_hidden=True)   # (B, n_bars, 16)
        return h

    def forward(self, x: torch.Tensor, chord_tokens: torch.Tensor) -> torch.Tensor:
        """
        x            : (B, seq_len, subseq_len)  preprocessed CP tokens
        chord_tokens : (B, n_bars)               bar-level chord token indices
        Returns logits of shape (B, seq_len, subseq_len, vocab_size) via local_decode.
        """
        base = self.base
        batch_size, seq_len, _ = x.shape

        rule_hidden = self._rule_hidden(chord_tokens)   # (B, n_bars, 16)

        h, emb = base.local_encode(x)
        h = h.view(batch_size, seq_len, base.hidden_size)
        sos = base.global_sos.view(1, 1, -1).expand(batch_size, 1, -1)
        h = torch.cat([sos, h[:, :-1]], dim=1)              # (B, seq_len, hidden)

        mask = base.buffered_future_mask(h)
        bar_indices = torch.arange(seq_len, device=h.device) // self.subbeats_per_bar

        for i, layer in enumerate(base.model.layer):
            h = layer(h, attention_mask=mask)[0]
            if (i + 1) % self.n_skip == 0:
                adapter_idx = (i + 1) // self.n_skip - 1
                # sub_offset=0: absolute subbeat positions start at 0
                h = h + self.yinyang_attn[adapter_idx](h, rule_hidden, sub_offset=0)

        return base.local_decode(h, emb)

    def loss(self, x: torch.Tensor, pitch_shift: torch.Tensor,
             chord_tokens: torch.Tensor) -> torch.Tensor:
        x_proc = self.base.preprocess(x, pitch_shift)
        logits = self(x_proc, chord_tokens)
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
    def global_sampling_chord(
        self,
        x:            torch.Tensor,   # (B, seed_len, subseq_len)  seed subbeats
        chord_tokens: torch.Tensor,   # (B, n_bars)
        max_seq_len:  int   = 384,
        temperature:  float = 1.0,
        sampling_func = None,
    ) -> list:
        base = self.base
        batch_size, seed_len, _ = x.shape

        rule_hidden = self._rule_hidden(chord_tokens)  # (B, n_bars, 16)

        # Encode seed subbeats
        h_seed, _ = base.local_encode(x)
        h_seed = h_seed.view(batch_size, seed_len, base.hidden_size)
        sos = base.global_sos.view(1, 1, -1).expand(batch_size, 1, -1)
        h = torch.cat([sos, h_seed], dim=1)                # (B, seed_len+1, hidden)

        y = [x[:, i, :] for i in range(seed_len)]
        past_key_values = None
        h_next = h                                         # first step: full seed

        for i in range(seed_len, max_seq_len):
            if i % 10 == 0:
                print(f'Sampling {i}/{max_seq_len}')

            cur_len = h_next.shape[1]
            attn_mask = base.buffered_future_mask(h) if past_key_values is None else None

            # --- global transformer (with KV cache) ---
            # Run each layer manually so we can inject the adapter
            # at the last position after the final layer.
            # KV caching only works cleanly with single-point injection (after last layer).
            h_out = h_next
            new_kvs = []
            for j, layer in enumerate(base.model.layer):
                past_kv_j = past_key_values[j] if past_key_values is not None else None
                layer_out = layer(
                    h_out,
                    attention_mask=attn_mask,
                    past_key_value=past_kv_j,
                    use_cache=True,
                )
                h_out = layer_out[0]
                new_kvs.append(layer_out[1])

                # Inject adapter after every n_skip layers, at last position only
                if (j + 1) % self.n_skip == 0:
                    adapter_idx = (j + 1) // self.n_skip - 1
                    # Query is only the last position; sub_offset = i (current step)
                    correction = self.yinyang_attn[adapter_idx](
                        h_out[:, -1:, :], rule_hidden, sub_offset=i
                    )
                    h_out = torch.cat([h_out[:, :-1, :], h_out[:, -1:, :] + correction], dim=1)

            past_key_values = new_kvs

            # Sample next subbeat from last global hidden
            y_next = base.local_sampling(
                h_out[:, -1], temperature=temperature,
                global_step=i, sampling_func=sampling_func,
            )
            y.append(y_next)

            # Encode next subbeat for next step
            h_next = base.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
            if past_key_values is None:
                h = torch.cat([h, h_next], dim=1)   # grow h for attention mask

        return y
