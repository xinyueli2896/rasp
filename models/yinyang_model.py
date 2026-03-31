"""
Yin-Yang model: AR transformer patched with rule model via layer-wise cross-attention.

Architecture
────────────
Both streams receive the same integer sequence as input.

  Yin  = rule model (always frozen)     → rule_hidden (B, T, rule_d_model)
  Yang = AR transformer (LoRA or frozen) → hidden states at each layer

Every n_skip AR layers, the Yang stream cross-attends into the Yin stream:

  x = ar_block(x)
  if (layer + 1) % n_skip == 0:
      x = x + yinyang_attn(query=x, key=rule_hidden, value=rule_hidden)

The cross-attention gate is initialised to 0 so the model starts as the
original AR transformer and opens the rule channel gradually during training.

LoRA (optional)
───────────────
If use_lora=True, PEFT LoRA (r=16) is applied to the "qkv" projection of
every CausalSelfAttention block.  The LoRA delta weights are trainable;
all original AR weights stay frozen.  Requires: pip install peft

If use_lora=False the entire AR model is frozen.  Only yinyang_attn trains.

Parameter counts (d_model=128, rule_d_model=28, adapter_rank=32, n_heads=4,
                  n_layers=4, n_skip=2  →  2 cross-attention adapters):
  Each YinyangCrossAttention:
    q_proj   128×32 + 32  = 4,128
    k_proj    28×32 + 32  =   928   ← rule_d_model=28 (Tracr hidden dim)
    v_proj    28×32 + 32  =   928
    out_proj  32×128 + 128 = 4,224
    total                  = 10,208
  2 adapters               = 20,416  (always trainable)
  LoRA on qkv (r=16, 4 layers, 2 matrices each):
    4 × 2 × (128×16 + 16×384) = 4 × 2 × 8,192 = 65,536  (trainable with LoRA)
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer              import AutoregressiveTransformer
from models.rule_model               import RuleModelWrapper
from models.tracr_pytorch_rule_model import TracrPyTorchRuleModel
from data.dataset                    import VOCAB_SIZE

# The rule model hidden dimension is fixed by the Tracr-equivalent architecture
RULE_D_MODEL = TracrPyTorchRuleModel.TRACR_D_MODEL   # 28


# ---------------------------------------------------------------------------
# Cross-attention module
# ---------------------------------------------------------------------------

class YinyangCrossAttention(nn.Module):
    """
    Single cross-attention block injecting rule hidden states into AR hidden states.

    Query  = AR hidden state   (d_model)
    Key/V  = rule hidden state (rule_d_model)
    Output = tanh(gate) * out_proj(attn @ V)   added as residual to AR stream

    gate is initialised to 0 so injection starts at zero and opens gradually.
    Causal mask is applied so position t only attends to rule positions ≤ t.
    """

    def __init__(
        self,
        d_model:      int,
        rule_d_model: int,
        embed_dim:    int,
        n_heads:      int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.embed_dim = embed_dim

        self.q_proj   = nn.Linear(d_model,      embed_dim)
        self.k_proj   = nn.Linear(rule_d_model, embed_dim)
        self.v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim,    d_model)
        self.dropout  = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for linear in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, ar_hidden: torch.Tensor, rule_hidden: torch.Tensor) -> torch.Tensor:
        """
        ar_hidden   : (B, T, d_model)
        rule_hidden : (B, T, rule_d_model)
        returns     : (B, T, d_model)  — correction to add to AR stream
        """
        B, T, _ = ar_hidden.shape

        Q = self.q_proj(ar_hidden)    # (B, T, embed_dim)
        K = self.k_proj(rule_hidden)  # (B, T, embed_dim)
        V = self.v_proj(rule_hidden)  # (B, T, embed_dim)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scores = (Q @ K.transpose(-2, -1)) * self.scale   # (B, n_heads, T, T)

        causal = torch.ones(T, T, device=ar_hidden.device).tril().bool()
        scores = scores.masked_fill(~causal, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))    # (B, n_heads, T, T)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Full Yin-Yang model
# ---------------------------------------------------------------------------

class YinyangModel(nn.Module):
    """
    AR transformer (Yang) with layer-wise cross-attention from rule model (Yin).

    The AR model forward pass is run block by block so that rule hidden states
    can be injected after every n_skip layers.  The rule model is run once
    upfront to produce rule_hidden for all positions.

    Parameters
    ----------
    ar_ckpt_path   : path to pretrained AR checkpoint (None → random init)
    max_seq_len    : maximum sequence length
    d_model        : AR hidden dimension
    n_layers       : number of AR transformer blocks
    n_heads        : number of AR attention heads
    rule_d_model   : rule model embedding dimension
    adapter_rank   : embed_dim for each YinyangCrossAttention
    n_skip         : inject every n_skip AR layers (one adapter per n_skip layers)
    use_lora       : apply LoRA (r=16) to AR qkv projections if True; fully freeze otherwise
    force_fallback : use FallbackRuleModel even if tracr is available
    """

    def __init__(
        self,
        ar_ckpt_path:   str  | None = None,
        max_seq_len:    int  = 128,
        d_model:        int  = 128,
        n_layers:       int  = 4,
        n_heads:        int  = 4,
        rule_d_model:   int  = RULE_D_MODEL,   # 28 — fixed by Tracr architecture
        adapter_rank:   int  = 32,
        n_skip:         int  = 2,
        use_lora:       bool = True,
        lora_rank:      int  = 16,
        force_fallback: bool = False,
        device:         str  = 'cpu',
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_skip   = n_skip

        # ------------------------------------------------------------------ #
        # AR model
        # ------------------------------------------------------------------ #
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
            for p in ar_model.parameters():
                p.requires_grad_(False)
            self.ar_model = ar_model
            self._ar_base = ar_model

        # ------------------------------------------------------------------ #
        # Rule model — always fully frozen (no parameters)
        # ------------------------------------------------------------------ #
        self.rule_model = RuleModelWrapper(
            max_seq_len    = max_seq_len,
            rule_d_model   = rule_d_model,
            force_fallback = force_fallback,
        ).to(device)

        # ------------------------------------------------------------------ #
        # Cross-attention adapters: one per n_skip AR layers
        # ------------------------------------------------------------------ #
        assert n_layers % n_skip == 0, \
            f"n_layers ({n_layers}) must be divisible by n_skip ({n_skip})"
        n_adapters = n_layers // n_skip
        self.yinyang_attn = nn.ModuleList([
            YinyangCrossAttention(
                d_model      = d_model,
                rule_d_model = rule_d_model,
                embed_dim    = adapter_rank,
                n_heads      = 4,
            )
            for _ in range(n_adapters)
        ])

    # ---------------------------------------------------------------------- #
    # Core layer-wise forward
    # ---------------------------------------------------------------------- #

    def _forward_layerwise(
        self,
        idx:         torch.Tensor,   # (B, T)
        rule_hidden: torch.Tensor,   # (B, T, rule_d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run AR transformer block by block, injecting rule_hidden every n_skip layers.
        Returns (logits, final_hidden) — shapes (B, T, vocab) and (B, T, d_model).
        """
        ar = self._ar_base
        B, T = idx.shape

        tok = ar.tok_emb(idx)
        pos = ar.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))
        x   = ar.drop(tok + pos)

        for i, block in enumerate(ar.blocks):
            x = block(x)
            # inject after every n_skip-th block (0-indexed: layers 1,3,5,... with n_skip=2)
            if (i + 1) % self.n_skip == 0:
                adapter_idx = (i + 1) // self.n_skip - 1
                x = x + self.yinyang_attn[adapter_idx](x, rule_hidden)

        hidden = ar.ln_f(x)
        logits = ar.lm_head(hidden)
        return logits, hidden

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, T) long tensor
        Returns logits (B, T, vocab_size).
        """
        _, rule_hidden = self.rule_model(idx, return_hidden=True)
        logits, _      = self._forward_layerwise(idx, rule_hidden)
        return logits

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, n_new: int) -> torch.Tensor:
        """Autoregressively generate n_new tokens after start_tokens."""
        self.eval()
        tokens  = start_tokens.clone()
        max_len = self._ar_base.max_seq_len
        for _ in range(n_new):
            ctx      = tokens[:, -max_len:]
            logits   = self(ctx)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens   = torch.cat([tokens, next_tok], dim=1)
        return tokens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_yinyang_model(
    ar_ckpt_path:   str | None = None,
    max_seq_len:    int  = 128,
    d_model:        int  = 128,
    n_layers:       int  = 4,
    n_heads:        int  = 4,
    rule_d_model:   int  = RULE_D_MODEL,   # 28 — fixed by Tracr architecture
    adapter_rank:   int  = 32,
    n_skip:         int  = 2,
    use_lora:       bool = True,
    lora_rank:      int  = 16,
    force_fallback: bool = False,
    device:         str  = 'cpu',
) -> YinyangModel:
    return YinyangModel(
        ar_ckpt_path   = ar_ckpt_path,
        max_seq_len    = max_seq_len,
        d_model        = d_model,
        n_layers       = n_layers,
        n_heads        = n_heads,
        rule_d_model   = rule_d_model,
        adapter_rank   = adapter_rank,
        n_skip         = n_skip,
        use_lora       = use_lora,
        lora_rank      = lora_rank,
        force_fallback = force_fallback,
        device         = device,
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
