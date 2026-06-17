
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import AutoregressiveTransformer
from models.rule_model  import RuleModelWrapper
from data.dataset       import VOCAB_SIZE

# rule_hidden is the next-token distribution: shape (B, T, VOCAB_SIZE)
RULE_D_MODEL = VOCAB_SIZE   # 12


class YinyangCrossAttention(nn.Module):
    # query=AR hidden, key/value=rule hidden, causal mask, output scaled by learnable gate
    #
    # Q and K are BOTH PURELY POSITIONAL (scaled sin/cos, no learned projections).
    # rule_hidden[q] = one_hot(next_token[q]), shape (B, T, 12).
    # Scaling pe by PE_SCALE=20 gives diagonal Q·K advantage of ~70 per score unit,
    # making attention nearly one-hot on the same position. V = v_proj(rule_hidden)
    # extracts the next-token signal, which out_proj injects into the AR residual stream.

    PE_SCALE = 20.0   # makes diagonal Q·K score >> off-diagonal; matches Tracr attn_scale

    def __init__(
        self,
        d_model:      int,
        rule_d_model: int,
        embed_dim:    int,
        n_heads:      int = 4,
        dropout:      float = 0.1,
        max_seq_len:  int = 128,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.embed_dim = embed_dim

        # No q_proj or k_proj: both Q and K are purely positional.
        # V is the only content-bearing projection.
        self.v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim,    d_model)

        # gate=1 at init: full contribution from start, can scale up/down during training
        # (do NOT init to 0 — with frozen AR, gate=0 kills gradients for all adapter weights)
        self.gate      = nn.Parameter(torch.ones(1))
        self.attn_drop = nn.Dropout(dropout)

        # Sinusoidal pos encoding scaled by PE_SCALE.
        # score(q,k) = PE_SCALE² * (pe[q]·pe[k]) / sqrt(head_dim)
        # Diagonal advantage per adjacent pair ≈ PE_SCALE² * 0.5 / sqrt(head_dim) ≈ 70.
        import math
        pe  = torch.zeros(max_seq_len, embed_dim)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe * self.PE_SCALE
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for linear in [self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, ar_hidden: torch.Tensor, rule_hidden: torch.Tensor,
                indices_query: torch.Tensor | None = None,
                indices_key:   torch.Tensor | None = None) -> torch.Tensor:
        B, T_q, _ = ar_hidden.shape
        B, T_k, _ = rule_hidden.shape

        if indices_query is None:
            indices_query = torch.arange(T_q, device=ar_hidden.device)
        if indices_key is None:
            indices_key = torch.arange(T_k, device=ar_hidden.device)

        pos_q = self.pos_enc[:, indices_query, :]                          # (1, T_q, embed_dim)
        pos_k = self.pos_enc[:, indices_key,   :]                          # (1, T_k, embed_dim)
        Q = pos_q.expand(B, -1, -1)                                        # (B, T_q, embed_dim) — purely positional
        K = pos_k.expand(B, -1, -1)                                        # (B, T_k, embed_dim) — purely positional
        V = self.v_proj(rule_hidden)                                       # (B, T_k, embed_dim)

        Q = Q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)       # (B, n_heads, T_q, T_k)

        causal = torch.ones(T_q, T_k, device=ar_hidden.device).tril().bool()
        scores = scores.masked_fill(~causal, float('-inf'))

        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(out) * self.gate


class LearnedRuleInputEncoder(nn.Module):
    """
    Replaces W_E[tokens] (the hard one-hot token embedding) as the input to
    the rule model's frozen attention block.

    encoder_type='embedding'  : plain token embedding lookup (no context).
    encoder_type='transformer': embedding + bidirectional TransformerEncoder.
    encoder_type='softmax'    : position-separate 4×V×V tables. Cannot generalise
                                to token values unseen at a given position class.
    encoder_type='mlp'        : concat(one_hot(token), one_hot(pos%4)) → fc1 →
                                ReLU → fc2 → softmax. Memorises training pairs;
                                does NOT generalise to unseen (token, pos_class) pairs.
    encoder_type='additive'   : logits = W_tok[token] + W_pos[pos_class].
                                Forces factorised computation: W_tok learns per-token
                                direction, W_pos learns per-pos-class shift.
                                All 12 token values and all 4 pos_classes are seen
                                during adapter training (at different combinations),
                                so W_tok and W_pos are fully trained. Their sum
                                generalises to unseen (token, pos_class) pairs.
    encoder_type='circular'   : circular convolution encoder.
                                shift_logits[p] ∈ R^V represents the shift distribution
                                for pos_class p.  For output token t' given input token t:
                                  logit[t'] = shift_logits[pos_class, (t'-t) % V]
                                Because the shift is purely a function of pos_class (not
                                the token), all training pairs at pos_class p share the
                                same ground-truth shift → overdetermined → unique
                                convergence → generalises to ANY input token including
                                unseen starters.
    """

    def __init__(
        self,
        vocab_size:    int,
        rule_d_model:  int   = RULE_D_MODEL,
        encoder_type:  str   = 'embedding',   # 'embedding'|'transformer'|'softmax'|'mlp'|'additive'
        n_layers:      int   = 2,
        n_heads:       int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.vocab_size   = vocab_size
        self.rule_d_model = rule_d_model

        if encoder_type == 'softmax':
            # Position-separate: 4 independent V×V logit tables.
            self.token_logits = nn.Parameter(
                torch.eye(vocab_size).unsqueeze(0).expand(4, -1, -1).clone() * 10.0
            )
        elif encoder_type == 'mlp':
            # concat(one_hot(token), one_hot(pos%4)) → fc1 → ReLU → fc2 → V-dim probs.
            hidden = 128
            self.mlp_fc1 = nn.Linear(vocab_size + 4, hidden)
            self.mlp_fc2 = nn.Linear(hidden, vocab_size)
            nn.init.xavier_uniform_(self.mlp_fc1.weight)
            nn.init.zeros_(self.mlp_fc1.bias)
            nn.init.xavier_uniform_(self.mlp_fc2.weight)
            nn.init.zeros_(self.mlp_fc2.bias)
        elif encoder_type == 'additive':
            self.W_tok = nn.Parameter(torch.zeros(vocab_size, vocab_size))
            self.W_pos = nn.Parameter(torch.zeros(4, vocab_size))
            nn.init.normal_(self.W_tok, std=0.01)
            nn.init.normal_(self.W_pos, std=0.01)
        elif encoder_type == 'fourier':
            # Fourier-rotation encoder: cyclic shift IS a rotation in Fourier space.
            # logits = (R[pos_class] @ fourier(token)) · fourier(t') for each t'
            # R[p] ∈ R^(V×V) is a learned rotation matrix per pos_class.
            # KEY: R[p] is token-independent, so it generalises to ANY token once
            # trained — even tokens unseen at pos_class p in training.
            # All V=12 token values appear in training at SOME pos_class, so fourier(t)
            # is meaningful for every t. All 4 pos_classes appear in training, so R[p]
            # is fully learned for every p.
            import math
            V = vocab_size
            t_idx = torch.arange(V).float()
            cols  = [torch.ones(V)]   # DC (k=0)
            for k in range(1, V // 2):
                cols.append(torch.cos(2 * math.pi * k * t_idx / V))
                cols.append(torch.sin(2 * math.pi * k * t_idx / V))
            cols.append(torch.cos(math.pi * t_idx))  # Nyquist (k=V/2)
            fourier_feats = torch.stack(cols, dim=1)  # (V, V)
            self.register_buffer('fourier_feats', fourier_feats)
            # Init to identity: no rotation at start, training learns the right shift
            self.R = nn.Parameter(torch.eye(V).unsqueeze(0).expand(4, -1, -1).clone())
        elif encoder_type == 'circular':
            # shift_logits[p, s] = unnorm log-prob that shift at pos_class p equals s.
            # Warm-start: expected shifts are pc0→5, pc1→2, pc2→5, pc3→0
            self.shift_logits = nn.Parameter(torch.zeros(4, vocab_size))
            with torch.no_grad():
                for p, s in enumerate([5, 2, 5, 0]):
                    self.shift_logits[p, s] = 4.0
        else:
            self.embedding = nn.Embedding(vocab_size, rule_d_model)

        if encoder_type == 'transformer':
            import warnings
            encoder_layer = nn.TransformerEncoderLayer(
                d_model         = rule_d_model,
                nhead           = n_heads,
                dim_feedforward = rule_d_model * 4,
                dropout         = dropout,
                batch_first     = True,
                norm_first      = True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
                )
        elif encoder_type not in ('embedding', 'softmax', 'mlp', 'additive', 'fourier', 'circular'):
            raise ValueError(
                f"encoder_type must be one of 'embedding', 'transformer', 'softmax', 'mlp', "
                f"'additive', 'fourier', 'circular', got {encoder_type!r}"
            )

    def forward(self, tokens: torch.Tensor,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (B, T), positions: (T,) sequence indices
        if self.encoder_type == 'softmax':
            B, T = tokens.shape
            if positions is not None:
                pos_class = (positions % 4).long()
                p = pos_class.unsqueeze(0).expand(B, -1)
                logits = self.token_logits[p, tokens]
            else:
                logits = self.token_logits[0][tokens]
            probs = F.softmax(logits, dim=-1)
            pad   = torch.zeros(*probs.shape[:-1], self.rule_d_model - self.vocab_size,
                                device=tokens.device)
            return torch.cat([probs, pad], dim=-1)

        if self.encoder_type == 'mlp':
            B, T = tokens.shape
            tok_oh = F.one_hot(tokens, num_classes=self.vocab_size).float()   # (B, T, V)
            if positions is not None:
                pos_class = (positions % 4).long()                            # (T,)
                pos_oh = F.one_hot(pos_class, num_classes=4).float()          # (T, 4)
                pos_oh = pos_oh.unsqueeze(0).expand(B, -1, -1)               # (B, T, 4)
            else:
                pos_oh = torch.zeros(B, T, 4, device=tokens.device)
            x      = torch.cat([tok_oh, pos_oh], dim=-1)                     # (B, T, V+4)
            h      = F.relu(self.mlp_fc1(x))                                 # (B, T, 2V)
            logits = self.mlp_fc2(h)                                          # (B, T, V)
            probs  = F.softmax(logits, dim=-1)
            pad    = torch.zeros(*probs.shape[:-1], self.rule_d_model - self.vocab_size,
                                 device=tokens.device)
            return torch.cat([probs, pad], dim=-1)

        if self.encoder_type == 'additive':
            B, T = tokens.shape
            if positions is not None:
                pos_class = (positions % 4).long()
            else:
                pos_class = torch.zeros(T, dtype=torch.long, device=tokens.device)
            tok_logits = self.W_tok[tokens]
            pos_logits = self.W_pos[pos_class].unsqueeze(0).expand(B, -1, -1)
            logits = tok_logits + pos_logits
            probs  = F.softmax(logits, dim=-1)
            pad    = torch.zeros(*probs.shape[:-1], self.rule_d_model - self.vocab_size,
                                 device=tokens.device)
            return torch.cat([probs, pad], dim=-1)

        if self.encoder_type == 'fourier':
            B, T = tokens.shape
            if positions is not None:
                pos_class = (positions % 4).long()
            else:
                pos_class = torch.zeros(T, dtype=torch.long, device=tokens.device)
            phi   = self.fourier_feats[tokens]                                # (B, T, V)
            R_p   = self.R[pos_class]                                         # (T, V, V)
            v     = torch.einsum('bti,tij->btj', phi, R_p)                   # (B, T, V)
            logits = v @ self.fourier_feats.t()                               # (B, T, V)
            probs  = F.softmax(logits, dim=-1)
            pad    = torch.zeros(*probs.shape[:-1], self.rule_d_model - self.vocab_size,
                                 device=tokens.device)
            return torch.cat([probs, pad], dim=-1)

        if self.encoder_type == 'circular':
            B, T = tokens.shape
            V = self.vocab_size
            if positions is not None:
                pos_class = (positions % 4).long()
            else:
                pos_class = torch.zeros(T, dtype=torch.long, device=tokens.device)
            shift_p = self.shift_logits[pos_class]               # (T, V)
            # logit for output t' given token t: shift_logits[pc, (t'-t) % V]
            idx_v = torch.arange(V, device=tokens.device)
            gather_idx = (idx_v[None, None, :] - tokens[:, :, None]) % V    # (B, T, V)
            logits = shift_p[None, :, :].expand(B, -1, -1).gather(2, gather_idx)  # (B, T, V)
            probs  = F.softmax(logits, dim=-1)
            pad    = torch.zeros(*probs.shape[:-1], self.rule_d_model - self.vocab_size,
                                 device=tokens.device)
            return torch.cat([probs, pad], dim=-1)

        h = self.embedding(tokens)
        if self.encoder_type == 'transformer':
            h = self.transformer(h)
        return h


class BidirectionalYinyangAttention(nn.Module):
    """
    Bidirectional cross-attention where AR provides proxy input to rule model.

    Step 1 (AR → rule):
      AR hidden states are projected into rule space and augmented with the
      frozen positional embeddings W_pos[q%4]. This proxy is then processed
      by the rule model's own frozen attention matrices W_Q, W_K, W_V, W_O.
      No token input to the rule model — AR's representation substitutes.
      Because W_Q/W_K zero out the token dims of the proxy, the attention
      pattern is purely positional (identical to the Tracr rule model).
      W_V reads the token-subspace of the proxy — i.e. what AR put there.

    Step 2 (rule → AR):
      AR cross-attends to the rule model's output (causal mask, same as
      the unidirectional YinyangCrossAttention).
    """

    def __init__(
        self,
        d_model:      int,
        rule_d_model: int,
        embed_dim:    int,
        n_heads:      int   = 4,
        dropout:      float = 0.1,
        max_seq_len:  int   = 128,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads          = n_heads
        self.head_dim         = embed_dim // n_heads
        self.embed_dim        = embed_dim
        self.rule_attn_scale  = 20.0   # matches Tracr rule model attn_scale

        # Step 1: project AR hidden states into rule space
        self.ar_to_rule = nn.Linear(d_model, rule_d_model)

        # Step 2: AR cross-attends to rule output (causal)
        self.fwd_q_proj   = nn.Linear(d_model,      embed_dim)
        self.fwd_k_proj   = nn.Linear(rule_d_model, embed_dim)
        self.fwd_v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.fwd_out_proj = nn.Linear(embed_dim,    d_model)

        self.gate      = nn.Parameter(torch.ones(1))
        self.attn_drop = nn.Dropout(dropout)

        import math
        pe  = torch.zeros(max_seq_len, embed_dim)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_enc', pe.unsqueeze(0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ar_to_rule.weight)
        nn.init.zeros_(self.ar_to_rule.bias)
        for m in [self.fwd_q_proj, self.fwd_k_proj, self.fwd_v_proj, self.fwd_out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _fwd_mha(self, Q, K, V) -> torch.Tensor:
        B, T_q, _ = Q.shape
        _, T_k, _ = K.shape
        Q = Q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask   = torch.ones(T_q, T_k, device=Q.device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn   = self.attn_drop(F.softmax(scores, dim=-1))
        return (attn @ V).transpose(1, 2).contiguous().view(B, T_q, self.n_heads * self.head_dim)

    def forward(self, ar_hidden: torch.Tensor,
                W_Q: torch.Tensor, W_K: torch.Tensor,
                W_V: torch.Tensor, W_O: torch.Tensor) -> torch.Tensor:
        """
        ar_hidden : (B, T, d_model)
        W_Q/K/V/O : (28, 28) — frozen rule model attention matrices
        """
        B, T, _ = ar_hidden.shape

        # --- Step 1: AR → rule ---
        # ar_to_rule learns the full d_model→rule_d_model mapping.
        # AR hidden states already encode position; the linear can learn to
        # populate the rule model's positional subspace (dims 24-27) from that.
        proxy = self.ar_to_rule(ar_hidden)                         # (B, T, 28)

        # Run rule model's frozen attention on proxy (no causal mask)
        Q_rule   = proxy @ W_Q.t()                                  # (B, T, 28)
        K_rule   = proxy @ W_K.t()                                  # (B, T, 28)
        V_rule   = proxy @ W_V.t()                                  # (B, T, 28)
        scores   = (Q_rule @ K_rule.transpose(-1, -2)) * self.rule_attn_scale
        attn     = F.softmax(scores, dim=-1)
        rule_out = (attn @ V_rule) @ W_O.t()                       # (B, T, 28)

        # --- Step 2: rule → AR ---
        pos        = self.pos_enc[:, :T, :]
        Q2         = self.fwd_q_proj(ar_hidden) + pos
        K2         = self.fwd_k_proj(rule_out)  + pos
        V2         = self.fwd_v_proj(rule_out)
        correction = self.fwd_out_proj(self._fwd_mha(Q2, K2, V2))  # (B, T, d_model)

        return correction * self.gate


# ---------------------------------------------------------------------------
# Full Yin-Yang model
# ---------------------------------------------------------------------------

class YinyangModel(nn.Module):

    def __init__(
        self,
        ar_ckpt_path:    str  | None = None,
        max_seq_len:     int  = 128,
        d_model:         int  = 128,
        n_layers:        int  = 4,
        n_heads:         int  = 4,
        rule_d_model:    int  = RULE_D_MODEL,   # 28 for frozen rule; free for encoder_mode
        adapter_rank:    int  = 32,
        n_skip:          int  = 2,
        use_lora:        bool = True,
        lora_rank:       int  = 16,
        force_fallback:  bool = False,
        device:          str  = 'cpu',
        train_ar:         bool = False,
        bidirectional:    bool = False,
        encoder_injected: bool = False,   # learned encoder replaces W_E before frozen W_Q/K/V/O
        encoder_type:     str  = 'embedding',  # 'embedding' | 'transformer'
        encoder_n_layers: int  = 2,
        encoder_n_heads:  int  = 4,
    ):
        super().__init__()
        self.n_layers         = n_layers
        self.n_skip           = n_skip
        self.bidirectional    = bidirectional
        self.encoder_injected = encoder_injected

        from data.dataset import VOCAB_SIZE as _VOCAB_SIZE
        ar_model = AutoregressiveTransformer(
            vocab_size  = _VOCAB_SIZE,
            max_seq_len = max_seq_len,
            d_model     = d_model,
            n_layers    = n_layers,
            n_heads     = n_heads,
        ).to(device)

        if ar_ckpt_path is not None:
            state = torch.load(ar_ckpt_path, map_location=device)
            ar_model.load_state_dict(state)
            print(f'[YinyangModel] Loaded AR weights from {ar_ckpt_path}')

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError(
                    "use_lora=True requires the 'peft' package: pip install peft"
                )
            lora_config = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_rank * 2,
                target_modules = ['qkv'],   # fused QKV linear in CausalSelfAttention
                lora_dropout   = 0.1,
                bias           = 'none',
            )
            self.ar_model = get_peft_model(ar_model, lora_config)
            # _ar_base gives direct access to tok_emb / blocks / ln_f / lm_head.
            # PEFT modifies qkv IN-PLACE inside the blocks, so running
            # _ar_base.blocks[i](x) already uses the LoRA-adapted qkv.
            self._ar_base = self.ar_model.base_model.model
        else:
            if not train_ar:
                for p in ar_model.parameters():
                    p.requires_grad_(False)
            self.ar_model = ar_model
            self._ar_base = ar_model

        self.rule_model = RuleModelWrapper(
            max_seq_len    = max_seq_len,
            rule_d_model   = rule_d_model,
            force_fallback = force_fallback,
        ).to(device)

        if encoder_injected:
            # Learned encoder replaces W_E[tokens] before the frozen W_Q/K/V/O block.
            # rule_d_model must stay 28 so shapes match.
            self.rule_input_encoder = LearnedRuleInputEncoder(
                vocab_size   = VOCAB_SIZE,
                rule_d_model = RULE_D_MODEL,
                encoder_type = encoder_type,
                n_layers     = encoder_n_layers,
                n_heads      = encoder_n_heads,
            ).to(device)
            # Initialize embedding to W_E so the pipeline is analytically correct
            # from epoch 0. Training refines rather than re-discovers the solution.
            with torch.no_grad():
                if encoder_type not in ('softmax', 'mlp', 'additive', 'fourier', 'circular'):
                    self.rule_input_encoder.embedding.weight.copy_(self._W_E)

        assert n_layers % n_skip == 0, \
            f"n_layers ({n_layers}) must be divisible by n_skip ({n_skip})"
        n_adapters  = n_layers // n_skip
        adapter_cls = BidirectionalYinyangAttention if bidirectional else YinyangCrossAttention
        self.yinyang_attn = nn.ModuleList([
            adapter_cls(
                d_model      = d_model,
                rule_d_model = rule_d_model,
                embed_dim    = adapter_rank,
                n_heads      = 4,
                max_seq_len  = max_seq_len,
            )
            for _ in range(n_adapters)
        ]).to(device)

    def _forward_layerwise(self, idx: torch.Tensor,
                           rule_hidden:   torch.Tensor | None = None,
                           indices_query: torch.Tensor | None = None,
                           indices_key:   torch.Tensor | None = None):
        ar = self._ar_base
        B, T = idx.shape

        tok = ar.tok_emb(idx)
        pos = ar.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))
        x   = ar.drop(tok + pos)

        for i, block in enumerate(ar.blocks):
            x = block(x)
            if (i + 1) % self.n_skip == 0:
                adapter_idx = (i + 1) // self.n_skip - 1
                if self.bidirectional:
                    x = x + self.yinyang_attn[adapter_idx](
                        x, self._W_Q, self._W_K, self._W_V, self._W_O
                    )
                else:
                    x = x + self.yinyang_attn[adapter_idx](x, rule_hidden,
                                                            indices_query=indices_query,
                                                            indices_key=indices_key)

        hidden = ar.ln_f(x)
        logits = ar.lm_head(hidden)
        return logits, hidden

    def _encoder_injected_rule_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Returns rule_hidden of shape (B, T, 12): the encoder's next-token
        probability distribution at each position, matching the format of the
        non-encoder-injected path (rule_hidden = one_hot(next_token)).
        """
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        enc_out = self.rule_input_encoder(idx, positions=pos)   # (B, T, 28 or 12)
        return enc_out[:, :, :VOCAB_SIZE]                       # (B, T, 12)

    def forward(self, idx: torch.Tensor,
                indices_query: torch.Tensor | None = None,
                indices_key:   torch.Tensor | None = None) -> torch.Tensor:
        if self.bidirectional:
            # AR proxy drives rule attention via frozen W_Q/K/V/O — no rule token input.
            rule_hidden = None
        elif self.encoder_injected:
            # Learned encoder replaces W_E before frozen rule attention.
            rule_hidden = self._encoder_injected_rule_hidden(idx)
        else:
            _, rule_hidden = self.rule_model(idx, return_hidden=True)
            rule_hidden    = rule_hidden.to(idx.device)
        logits, _ = self._forward_layerwise(idx, rule_hidden,
                                             indices_query=indices_query,
                                             indices_key=indices_key)
        return logits

    def get_encoder_logits(self, idx: torch.Tensor) -> torch.Tensor | None:
        """Pre-softmax logits (B, T, V) from the learned encoder for auxiliary supervision.

        Auxiliary CE loss should be against tgt (next tokens): the encoder must predict
        next_token[q] from (token[q], pos_class[q]). Direct supervision forces the
        encoder to learn the cyclic shift rule, enabling generalisation to unseen tokens.
        Returns None for encoder types without vocab-space logits.
        """
        if not (self.encoder_injected and hasattr(self, 'rule_input_encoder')):
            return None
        enc = self.rule_input_encoder
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        if enc.encoder_type == 'mlp':
            tok_oh    = F.one_hot(idx, num_classes=enc.vocab_size).float()
            pos_class = (pos % 4).long()
            pos_oh    = F.one_hot(pos_class, num_classes=4).float().unsqueeze(0).expand(B, -1, -1)
            x         = torch.cat([tok_oh, pos_oh], dim=-1)
            return enc.mlp_fc2(F.relu(enc.mlp_fc1(x)))           # (B, T, V) pre-softmax
        elif enc.encoder_type == 'softmax':
            pos_class = (pos % 4).long()
            p         = pos_class.unsqueeze(0).expand(B, -1)
            return enc.token_logits[p, idx]                       # (B, T, V) pre-softmax
        elif enc.encoder_type == 'additive':
            pos_class  = (pos % 4).long()
            tok_logits = enc.W_tok[idx]
            pos_logits = enc.W_pos[pos_class].unsqueeze(0).expand(B, -1, -1)
            return tok_logits + pos_logits
        elif enc.encoder_type == 'fourier':
            pos_class = (pos % 4).long()
            phi  = enc.fourier_feats[idx]                         # (B, T, V)
            R_p  = enc.R[pos_class]                               # (T, V, V)
            v    = torch.einsum('bti,tij->btj', phi, R_p)        # (B, T, V)
            return v @ enc.fourier_feats.t()                      # (B, T, V) pre-softmax
        elif enc.encoder_type == 'circular':
            V = enc.vocab_size
            pos_class = (pos % 4).long()
            shift_p   = enc.shift_logits[pos_class]               # (T, V)
            idx_v     = torch.arange(V, device=idx.device)
            gather_idx = (idx_v[None, None, :] - idx[:, :, None]) % V        # (B, T, V)
            return shift_p[None, :, :].expand(B, -1, -1).gather(2, gather_idx)  # (B, T, V)
        return None

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, n_new: int) -> torch.Tensor:
        self.eval()
        tokens  = start_tokens.clone()
        max_len = self._ar_base.max_seq_len
        for _ in range(n_new):
            ctx      = tokens[:, -max_len:]
            logits   = self(ctx)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens   = torch.cat([tokens, next_tok], dim=1)
        return tokens


def build_yinyang_model(
    ar_ckpt_path:    str | None = None,
    max_seq_len:     int  = 128,
    d_model:         int  = 128,
    n_layers:        int  = 4,
    n_heads:         int  = 4,
    rule_d_model:    int  = RULE_D_MODEL,   # 28 for frozen rule; free for encoder_mode
    adapter_rank:    int  = 32,
    n_skip:          int  = 2,
    use_lora:        bool = True,
    lora_rank:       int  = 16,
    force_fallback:  bool = False,
    device:          str  = 'cpu',
    train_ar:          bool = False,
    bidirectional:     bool = False,
    encoder_injected:  bool = False,
    encoder_type:      str  = 'embedding',
    encoder_n_layers:  int  = 2,
    encoder_n_heads:   int  = 4,
) -> YinyangModel:
    return YinyangModel(
        ar_ckpt_path     = ar_ckpt_path,
        max_seq_len      = max_seq_len,
        d_model          = d_model,
        n_layers         = n_layers,
        n_heads          = n_heads,
        rule_d_model     = rule_d_model,
        adapter_rank     = adapter_rank,
        n_skip           = n_skip,
        use_lora         = use_lora,
        lora_rank        = lora_rank,
        force_fallback   = force_fallback,
        device           = device,
        train_ar         = train_ar,
        bidirectional    = bidirectional,
        encoder_injected = encoder_injected,
        encoder_type     = encoder_type,
        encoder_n_layers = encoder_n_layers,
        encoder_n_heads  = encoder_n_heads,
    )


if __name__ == '__main__':
    model = build_yinyang_model(force_fallback=True, use_lora=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Trainable: {trainable:,}   Frozen: {frozen:,}')
    print(f'yinyang_attn modules: {len(model.yinyang_attn)}')
    for i, m in enumerate(model.yinyang_attn):
        n = sum(p.numel() for p in m.parameters())
        print(f'  adapter {i}: {n:,} params')

    x = torch.randint(0, 12, (2, 16))
    logits = model(x)
    print(f'output shape: {logits.shape}')   # (2, 16, 12)

    gen = model.generate(x[:, :1], n_new=8)
    print(f'generated shape: {gen.shape}')   # (2, 9)
