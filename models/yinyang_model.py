
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


class YinyangCrossAttention(nn.Module):
    # query=AR hidden, key/value=rule hidden, causal mask, output scaled by learnable gate
    # pos_proj removed: rule_hidden already encodes position in dims 24-27

    def __init__(
        self,
        d_model:      int,
        rule_d_model: int,
        embed_dim:    int,
        n_heads:      int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.q_proj   = nn.Linear(d_model,      embed_dim)
        self.k_proj   = nn.Linear(rule_d_model, embed_dim)
        self.v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim,    d_model)

        # gate=1 at init: full contribution from start, can scale up/down during training
        # (do NOT init to 0 — with frozen AR, gate=0 kills gradients for all adapter weights)
        self.gate      = nn.Parameter(torch.ones(1))
        self.attn_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for linear in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
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

        Q = self.q_proj(ar_hidden)                                         # (B, T_q, embed_dim)
        K = self.k_proj(rule_hidden)                                       # (B, T_k, embed_dim)
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


# ---------------------------------------------------------------------------
# Full Yin-Yang model
# ---------------------------------------------------------------------------

class YinyangModel(nn.Module):

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

        self.rule_model = RuleModelWrapper(
            max_seq_len    = max_seq_len,
            rule_d_model   = rule_d_model,
            force_fallback = force_fallback,
        ).to(device)

        from data.dataset import VOCAB_SIZE as _VOCAB_SIZE2
        # Direct rule bypass: rule_hidden (dims 0-VOCAB_SIZE-1 = one-hot of next token)
        # → logit space. Near-identity is the optimal solution, making generalization easy.
        # Initialized small so early training is stable; gate grows as adapter learns.
        self.rule_bypass      = nn.Linear(rule_d_model, _VOCAB_SIZE2, bias=False).to(device)
        self.rule_bypass_gate = nn.Parameter(torch.zeros(1).to(device))  # starts at 0, grows
        nn.init.zeros_(self.rule_bypass.weight)

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
        ]).to(device)

    def _forward_layerwise(self, idx: torch.Tensor, rule_hidden: torch.Tensor,
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
                x = x + self.yinyang_attn[adapter_idx](x, rule_hidden,
                                                        indices_query=indices_query,
                                                        indices_key=indices_key)

        hidden = ar.ln_f(x)
        logits = ar.lm_head(hidden)
        return logits, hidden

    def forward(self, idx: torch.Tensor,
                indices_query: torch.Tensor | None = None,
                indices_key:   torch.Tensor | None = None) -> torch.Tensor:
        _, rule_hidden = self.rule_model(idx, return_hidden=True)
        rule_hidden    = rule_hidden.to(idx.device)
        logits, _      = self._forward_layerwise(idx, rule_hidden,
                                                  indices_query=indices_query,
                                                  indices_key=indices_key)
        # Direct rule bypass: rule_hidden[:,:,:VOCAB_SIZE] is one-hot of predicted next token.
        # This shortcut lets the adapter learn near-identity mapping without routing through
        # the AR residual stream, greatly improving generalization to unseen starters.
        bypass = self.rule_bypass(rule_hidden) * self.rule_bypass_gate   # (B, T, VOCAB_SIZE)
        return logits + bypass

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
