"""
CP-Transformer Yin-Yang adapter for rule-conditioned bass generation.

Architecture
------------
  BassTracrRuleModel  (models/bass_tracr_rule_model.py)
    Zero-parameter analytical model. Reads starting pitch class from the
    first beat of the sequence (x = pc[0]) and encodes:
      hidden[t] = one-hot((x + OFFSETS[t%4]) % 12) + pos_encoding[t%4]
    i.e. the CURRENT pitch class at position t.
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
      rule_hidden[t] = embed(key + OFFSETS[t%4]) + pos[t%4]. The adapter learns
      to attend rule_hidden[0] (= embed(key)) and uses query position encoding
      to derive the next note — routing that generalises to unseen keys.
"""

from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bass_tracr_rule_model import BassTracrRuleModel, CPChordRuleModel, TRACR_D_MODEL


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
# Learned rule-input encoder  (encoder-injected variant)
# ---------------------------------------------------------------------------

class BassRuleInputEncoder(nn.Module):
    """
    Replaces the frozen one-hot W_E[pc] lookup in BassTracrRuleModel with a
    learned embedding, keeping W_pos frozen.

    Standard analytical rule:
        rule_hidden[t] = W_E[(key + OFFSETS[t%4]) % 12]  +  W_pos[t%4]
                       = one_hot(cur_pc)                  +  pos_embed   (16-dim)

    Encoder-injected:
        rule_hidden[t] = encoder(cur_pc)  +  W_pos[t%4]
                         ^^^^^^^^^^^^^^
                         learned dense embedding instead of one-hot

    The learned embedding can generalise to unseen keys because all 12 pitch
    classes appear during training as intermediate notes in other keys' cycles.
    The cross-attention's k_proj/v_proj also benefit from a smoother input
    representation compared to sparse one-hots.
    """

    def __init__(self, n_pitches: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(n_pitches, n_pitches)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        return self.embedding(pc)   # (B, T, n_pitches)


class ChordEncoder(nn.Module):
    """
    Learned encoder for chord chromagram → rule-space embedding.

    Approach 2 (chord progression): at training time the observed chromagram
    (extracted from ALL voices at each timestep) is available as a paired input.
    The encoder maps it to the same 12-dim space that CPChordRuleModel uses so
    that the cross-attention adapter can learn chord-conditioned generation.

    Input : (B, T, 12) float binary chromagram
    Output: (B, T, 12) learned embedding  (dims 0-11 only; zero-padded externally)

    At inference, build_rule_hidden_analytic is used instead of the encoder,
    exactly like BassTracrRuleModel's analytical path.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(12, hidden)
        self.fc2 = nn.Linear(hidden, 12)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        """chroma: (B, T, 12)  →  (B, T, 12)"""
        return self.fc2(F.relu(self.fc1(chroma)))


class BassTokMlpEncoder(nn.Module):
    """
    Token-only MLP encoder: one_hot(pc, 12) → Linear(12→64) → ReLU → Linear(64→rule_d_model).
    No positional information — only the pitch class token is used as input.
    W_pos is still added in _encoder_injected_rule_hidden, keeping positional
    information in the frozen part of the residual stream.
    """

    def __init__(self, n_pitches: int = 12, hidden: int = 64):
        super().__init__()
        self.n_pitches = n_pitches
        self.fc1 = nn.Linear(n_pitches, hidden)
        self.fc2 = nn.Linear(hidden, n_pitches)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(pc, num_classes=self.n_pitches).float()
        return self.fc2(F.relu(self.fc1(x)))   # (B, T, n_pitches)


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
        adapter_rank:     int  = 256,
        n_skip:           int  = 4,
        lora_rank:        int  = 0,
        bidirectional:    bool = False,
        encoder_injected: bool = False,
        encoder_type:     str  = 'embedding',
        rule_mode:        str  = 'current',
        approach:         str  = 'bass',
    ):
        assert not (bidirectional and encoder_injected), \
            "bidirectional and encoder_injected are mutually exclusive"
        assert not (encoder_injected and rule_mode != 'current'), \
            "encoder_injected only supported with rule_mode='current'"
        assert encoder_type in ('embedding', 'token_mlp'), \
            f"encoder_type must be 'embedding' or 'token_mlp', got {encoder_type!r}"
        assert approach in ('bass', 'chord'), \
            f"approach must be 'bass' or 'chord', got {approach!r}"
        assert not (approach == 'chord' and rule_mode != 'current'), \
            "chord approach only supported with rule_mode='current'"
        super().__init__()
        self.base             = base_model
        self.n_skip           = n_skip
        self.lora_rank        = lora_rank
        self.bidirectional    = bidirectional
        self.encoder_injected = encoder_injected
        self.approach         = approach

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
        if approach == 'chord':
            self.rule_model = CPChordRuleModel(beats_per_bar=4)
        else:
            self.rule_model = BassTracrRuleModel(rule_mode=rule_mode)

        self.yinyang_attn = nn.ModuleList([
            CPYinyangCrossAttention(
                d_model      = self.base.hidden_size,
                rule_d_model = self.rule_model.d_model,
                embed_dim    = adapter_rank,
                n_heads      = 8,
            )
            for _ in range(n_adapters)
        ])

        if encoder_injected:
            # Learned encoder replaces the analytical rule model's embedding lookup.
            # For bass approach: encodes pitch class → 12-dim embedding.
            # For chord approach: encodes observed chromagram → 12-dim embedding.
            # Both output 12 dims only; _encoder_injected_rule_hidden zero-pads to d_model.
            if approach == 'chord':
                self.rule_input_encoder = ChordEncoder()
            elif encoder_type == 'token_mlp':
                self.rule_input_encoder = BassTokMlpEncoder()
            else:
                self.rule_input_encoder = BassRuleInputEncoder()

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

    def _cur_pc(self, x_proc: torch.Tensor) -> torch.Tensor:
        """Extract current pitch class at each position from preprocessed tokens."""
        B, T = x_proc.shape[0], x_proc.shape[1]
        key     = (x_proc[:, 0, 1] % 128 % 12).clamp(0, 11)   # (B,)
        t_idx   = torch.arange(T, device=x_proc.device)
        offsets = self.rule_model._offsets[t_idx % 4]           # (T,)
        return (key[:, None] + offsets[None, :]) % 12           # (B, T)

    def _extract_chromagram(self, x_proc: torch.Tensor) -> torch.Tensor:
        """Extract binary chromagram from all voices at each timestep.

        x_proc : (B, T, S)  preprocessed CP tokens (S = n_voices * 2)
            Slot 2v   = program token (0-127 for valid notes; eos/pad otherwise)
            Slot 2v+1 = pitch + (dur+1)*128 token; % 128 gives MIDI pitch

        Returns (B, T, 12) float binary chromagram.
        A pitch class is 1 if ANY valid voice plays that pc at that timestep.
        """
        B, T, S = x_proc.shape
        n_voices = S // 2
        pad_tok  = self.base.tokenizer.pad_token
        eos_tok  = self.base.tokenizer.eos_token
        device   = x_proc.device

        chroma = torch.zeros(B, T, 12, device=device, dtype=torch.float)
        for v in range(n_voices):
            prog_slot  = x_proc[:, :, 2 * v]          # (B, T)
            pitch_slot = x_proc[:, :, 2 * v + 1]      # (B, T)
            valid = (prog_slot != pad_tok) & (prog_slot != eos_tok)  # (B, T)
            pitch = pitch_slot % 128                   # (B, T) MIDI pitch 0-127
            pc    = (pitch % 12).clamp(0, 11)          # (B, T) pitch class
            pc_oh = F.one_hot(pc.long(), num_classes=12).float()     # (B, T, 12)
            chroma = torch.clamp(chroma + pc_oh * valid.unsqueeze(-1).float(), 0, 1)
        return chroma

    def _encoder_injected_rule_hidden(self, x_proc: torch.Tensor) -> torch.Tensor:
        """Encoder-injected rule_hidden: learned embedding for rule input.

        Bass approach  : encodes cur_pc[t] = (key + OFFSETS[t%4]) % 12
        Chord approach : encodes observed chromagram from all voices

        Returns encoder(input) zero-padded to d_model, plus frozen W_pos[t%4].
        """
        t_idx   = torch.arange(x_proc.shape[1], device=x_proc.device)
        d       = self.rule_model.d_model

        if self.approach == 'chord':
            chroma = self._extract_chromagram(x_proc)            # (B, T, 12)
            enc    = self.rule_input_encoder(chroma)             # (B, T, 12)
        else:
            cur_pc = self._cur_pc(x_proc)                        # (B, T)
            enc    = self.rule_input_encoder(cur_pc)             # (B, T, 12)

        h       = F.pad(enc, (0, d - enc.shape[-1]))             # (B, T, d_model)
        pos_emb = self.rule_model.W_pos[t_idx % 4]              # (T, d_model)
        return h + pos_emb.unsqueeze(0)

    def get_encoder_logits(self, x_proc: torch.Tensor):
        """Raw encoder output and its paired ground-truth target.

        Bass approach:
            Returns (enc_logits, cur_pc): (B,T,12), (B,T) long
            Supervised by CE loss — encoder should learn to reconstruct cur_pc.
        Chord approach:
            Returns (enc_logits, chromagram): (B,T,12), (B,T,12) float binary
            Supervised by BCE loss — encoder should reconstruct the chromagram.

        Returns None if encoder_injected is False.
        """
        if not self.encoder_injected:
            return None
        if self.approach == 'chord':
            chroma  = self._extract_chromagram(x_proc)            # (B, T, 12)
            enc_out = self.rule_input_encoder(chroma)             # (B, T, 12)
            return enc_out, chroma                                 # logits, BCE target
        else:
            cur_pc  = self._cur_pc(x_proc)                        # (B, T)
            enc_out = self.rule_input_encoder(cur_pc)             # (B, T, 12)
            return enc_out, cur_pc                                 # logits, CE target

    def _rule_hidden(self, x_proc: torch.Tensor) -> torch.Tensor:
        """Compute frozen analytical rule_hidden from preprocessed tokens.

        Reads key = pc[b, 0] from voice 0 (slot 1) and runs the rule model:
          bass  : BassTracrRuleModel  → (B, T, 16/28) with one-hot pitch encoding
          chord : CPChordRuleModel    → (B, T, 16) with binary chromagram encoding
        Both models encode (key, t) → expected rule signal at position t.
        """
        pc = x_proc[:, :, 1] % 128 % 12   # (B, T) voice-0 pitch class; pc[b,0] = key
        _, h = self.rule_model(pc, return_hidden=True)   # (B, T, d_model)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, subseq_len)  preprocessed CP tokens
        Returns logits of shape (B, seq_len, subseq_len, vocab_size) via local_decode.
        """
        base = self.base
        batch_size, seq_len, _ = x.shape

        if self.encoder_injected:
            rule_hidden = self._encoder_injected_rule_hidden(x)
        elif self.bidirectional:
            rule_hidden = None
        else:
            rule_hidden = self._rule_hidden(x)

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

        # Pre-compute rule_hidden for standard and encoder_injected modes.
        # Bidirectional computes rule_hidden on-the-fly from AR states each step.
        pc_0 = (x[:, 0, 1] % 128 % 12).clamp(0, 11)   # (B,) starting pitch class / key
        if self.encoder_injected and self.approach == 'bass':
            # Bass encoder: cur_pc is computable analytically → run learned encoder at inference.
            t_idx   = torch.arange(max_seq_len, device=x.device)
            offsets = self.rule_model._offsets[t_idx % 4]               # (max_seq_len,)
            cur_pc  = (pc_0[:, None] + offsets[None, :]) % 12           # (B, max_seq_len)
            d       = self.rule_model.d_model
            h_enc   = self.rule_input_encoder(cur_pc)                    # (B, max_seq_len, 12)
            h_enc   = F.pad(h_enc, (0, d - h_enc.shape[-1]))            # (B, max_seq_len, d)
            pos_emb = self.rule_model.W_pos[t_idx % 4]                  # (max_seq_len, d)
            rule_hidden = h_enc + pos_emb.unsqueeze(0)
        elif not self.bidirectional:
            # Analytical rule model (both bass non-encoder-injected, chord encoder-injected,
            # and chord non-encoder-injected).  Chord encoder is bypassed at inference since
            # the observed chromagram is not available; build_rule_hidden_analytic provides
            # the expected chromagram from the key analytically.
            rule_hidden = self.rule_model.build_rule_hidden_analytic(pc_0, max_seq_len, x.device)
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
