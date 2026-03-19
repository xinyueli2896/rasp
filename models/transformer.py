"""
Small autoregressive transformer (GPT-style) for next-token prediction
over a vocabulary of 12 integers.

Designed to be pretrained on a *subset* of starting integers so it has
a gap that the rule-model adapter can fill.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AutoregressiveTransformer(nn.Module):
    """
    Parameters
    ----------
    vocab_size   : int   – number of token types (12 for our task)
    max_seq_len  : int   – maximum sequence length
    d_model      : int   – hidden dimension
    n_layers     : int   – number of transformer layers
    n_heads      : int   – attention heads
    ffn_mult     : int   – FFN hidden = ffn_mult * d_model
    """

    def __init__(
        self,
        vocab_size:  int = 12,
        max_seq_len: int = 128,
        d_model:     int = 128,
        n_layers:    int = 4,
        n_heads:     int = 4,
        ffn_mult:    int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model     = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f   = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T) long tensor
        Returns logits (B, T, vocab_size) and optionally the final hidden state (B, T, d_model).
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence too long: {T} > {self.max_seq_len}"

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)          # (B, T, d_model)
        logits = self.lm_head(hidden)  # (B, T, vocab_size)

        if return_hidden:
            return logits, hidden
        return logits

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, n_new: int) -> torch.Tensor:
        """
        Autoregressively generate n_new tokens after start_tokens.
        start_tokens : (B, T0) long tensor
        Returns       : (B, T0 + n_new) long tensor
        """
        self.eval()
        tokens = start_tokens.clone()
        for _ in range(n_new):
            logits = self(tokens[:, -self.max_seq_len:])
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_transformer(**kwargs) -> AutoregressiveTransformer:
    return AutoregressiveTransformer(**kwargs)


if __name__ == "__main__":
    model = build_transformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randint(0, 12, (4, 23))
    logits, hidden = model(x, return_hidden=True)
    print(f"logits: {logits.shape}, hidden: {hidden.shape}")
