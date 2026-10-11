"""Cross-attention adapter: frozen AR + frozen compiled rule model.

    AR (frozen, 798K)  ──┐
                         ├─> every n_skip blocks:  h = h + gate * crossattn(h, r)
    rule model (frozen) ─┘

Only the cross-attention projections train. The AR is frozen so any change
in behaviour is attributable to the adapter, and the rule model is frozen
because it is a compiled program, not something to fit.

Q AND K ARE PURELY POSITIONAL
-----------------------------
Same design as models/yinyang_model.py's YinyangCrossAttention. Q and K are
scaled sinusoids (PE_SCALE=20) of the position alone -- no content enters
either. With the causal mask, the diagonal advantage makes AR position q
attend almost entirely to rule position q, so each position reads the rule
state aligned with it.

That alignment is load-bearing, not incidental. The chord experiment tried
putting music content in Q (`--content_residual q`) on the theory that
degenerate positional attention was a weakness; chord accuracy fell from
0.22 to 0.09. The hardcoded positional routing is doing real work.

V is the only content path: v_proj reads the rule hidden state, out_proj
writes a correction into the AR residual stream, scaled by a learned gate.

WHAT THE ADAPTER READS
----------------------
`IntegerRuleModel.program_hidden` -- 28 of the 75 residual dims:

    seed[12]        one-hot token[0], constant down the sequence
    next_phase[4]   one-hot (q+1) % 4
    next_tok[12]    one-hot of the answer for position q+1

NOT the full 75. Tracr keeps the token embedding alive end to end, so the
full stream carries `tokens[14]` -- a verbatim copy of the current token --
and `indices[32]`. Injecting those hands the adapter a shortcut that looks
like rule-reading and is not.

`rule_slice` chooses how much of the work the rule model does:

    'program'   seed + next_phase + next_tok   adapter just routes the answer
    'answer'    next_tok only                  same, minus the working
    'operands'  seed + next_phase              adapter must compute
                                               (s + OFFSETS[p]) % 12 itself

'operands' is the interesting one and the direct analogue of the chord
experiment's retrieve-vs-full split, but 'program' is the right first run:
if the adapter cannot route an answer handed to it in one-hot form, nothing
harder will work.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.integer_rule_model import IntegerRuleModel  # noqa: E402
from models.transformer import AutoregressiveTransformer  # noqa: E402

PE_SCALE = 20.0
SLICES = {
    'program':  ('seed', 'next_phase', 'next_tok'),
    'answer':   ('next_tok',),
    'operands': ('seed', 'next_phase'),
}


class RuleCrossAttention(nn.Module):
    """Positional Q/K, rule-hidden V, gated residual correction."""

    def __init__(self, d_model: int, rule_d_model: int, embed_dim: int,
                 n_heads: int = 4, dropout: float = 0.1,
                 max_seq_len: int = 128):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads, self.embed_dim = n_heads, embed_dim
        self.head_dim = embed_dim // n_heads

        self.v_proj   = nn.Linear(rule_d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim, d_model)
        # Init 1, not 0: the AR is frozen, so a zero gate would give the
        # adapter no gradient signal to escape from on the cells where the AR
        # is already wrong -- which is every cell that matters here.
        self.gate     = nn.Parameter(torch.ones(1))
        self.attn_drop = nn.Dropout(dropout)

        pe  = torch.zeros(max_seq_len, embed_dim)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_enc', (pe * PE_SCALE).unsqueeze(0))

        for lin in (self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, ar_hidden: torch.Tensor,
                rule_hidden: torch.Tensor) -> torch.Tensor:
        B, T_q, _ = ar_hidden.shape
        T_k = rule_hidden.shape[1]

        Q = self.pos_enc[:, :T_q].expand(B, -1, -1)
        K = self.pos_enc[:, :T_k].expand(B, -1, -1)
        V = self.v_proj(rule_hidden)

        sh = lambda t, T: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        Q, K, V = sh(Q, T_q), sh(K, T_k), sh(V, T_k)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal = torch.ones(T_q, T_k, device=ar_hidden.device).tril().bool()
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = (attn @ V).transpose(1, 2).reshape(B, T_q, self.embed_dim)
        return self.out_proj(out) * self.gate


class LoRALinear(nn.Module):
    """Frozen Linear + trainable low-rank update:  y = W0 x + b + (alpha/r) B A x.

    Self-contained rather than `peft`, which is an optional dependency the
    rest of this repo does not need for a 30-line update.

    B is zero-initialised, so at step 0 the module is EXACTLY the frozen
    layer and the AR starts from its pretrained behaviour rather than a
    perturbation of it. That matters for the forgetting comparison: any
    drift on pretrain_only has to come from training, not from init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for q in self.base.parameters():
            q.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def apply_lora(ar: AutoregressiveTransformer, rank: int,
               alpha: float | None = None,
               targets: tuple[str, ...] = ('attn.qkv', 'attn.proj')) -> int:
    """Wrap the targeted Linears in LoRALinear. Returns the trainable count.

    Defaults to the attention projections only, which is the usual choice and
    keeps the update small relative to the adapter it is being compared with.
    Pass targets=('attn.qkv', 'attn.proj', 'ffn.0', 'ffn.2') for everything.
    """
    n = 0
    for name, mod in list(ar.named_modules()):
        for child, sub in list(mod.named_children()):
            full = f'{name}.{child}' if name else child
            if isinstance(sub, nn.Linear) and any(t in full for t in targets):
                lora = LoRALinear(sub, rank, alpha)
                setattr(mod, child, lora)
                n += lora.A.numel() + lora.B.numel()
    return n


class IntegerRuleAdapter(nn.Module):
    """Frozen AR, frozen compiled rule model, trainable cross-attention."""

    def __init__(self, ar: AutoregressiveTransformer, rule: IntegerRuleModel,
                 n_skip: int = 1, embed_dim: int = 64, n_heads: int = 4,
                 dropout: float = 0.1, rule_slice: str = 'program',
                 train_ar: bool = False, lora_rank: int = 0,
                 lora_alpha: float | None = None,
                 rule_input: str = 'tokens', encoder_hidden: int = 0):
        super().__init__()
        if rule_slice not in SLICES:
            raise ValueError(f'rule_slice must be one of {list(SLICES)}')
        self.ar, self.rule = ar, rule
        self.n_skip, self.rule_slice = n_skip, rule_slice

        dims = []
        for nm in SLICES[rule_slice]:
            dims += rule.subspace(nm)
        self.register_buffer('rule_dims', torch.as_tensor(sorted(dims)))

        # 'tokens'  : the rule model is handed the input tokens one-hot.
        # 'encoder' : NO input. A learned encoder reads the AR stream BEFORE
        #             the transformer blocks -- i.e. drop(tok_emb + pos_emb)
        #             -- and its softmax output replaces the rule model's
        #             token embedding. This is the integer analogue of the
        #             chord experiment's --rule_from_layer 0, and of
        #             yinyang_model.py's `bidirectional` ar_to_rule, except
        #             that IntegerRuleModel.embed already accepts a
        #             distribution, so the encoder feeds it directly rather
        #             than assembling a residual stream by hand.
        if rule_input not in ('tokens', 'encoder'):
            raise ValueError("rule_input must be 'tokens' or 'encoder'")
        self.rule_input = rule_input
        self.encoder = None
        if rule_input == 'encoder':
            V = rule.vocab_size
            self.encoder = (
                nn.Sequential(nn.Linear(ar.d_model, encoder_hidden), nn.ReLU(),
                              nn.Linear(encoder_hidden, V))
                if encoder_hidden else nn.Linear(ar.d_model, V))
        # Stashed by rule_hidden() so the trainer can supervise the encoder
        # without a second forward pass.
        self._enc_logits = None

        n_ad = len(ar.blocks) // n_skip
        self.adapters = nn.ModuleList([
            RuleCrossAttention(ar.d_model, len(dims), embed_dim, n_heads,
                               dropout, ar.max_seq_len)
            for _ in range(n_ad)])

        if train_ar and lora_rank:
            raise ValueError('train_ar and lora_rank are alternatives: one '
                             'trains every AR weight, the other a low-rank '
                             'update to some of them.')
        self.train_ar, self.lora_rank = train_ar, lora_rank
        for q in self.ar.parameters():
            q.requires_grad_(train_ar)
        if lora_rank:
            # after the blanket freeze, so only the A/B factors are trainable
            apply_lora(self.ar, lora_rank, lora_alpha)
        # rule is frozen by construction: its weights are buffers and its
        # parameters() is empty, so there is nothing to switch off here.

    # -- introspection -----------------------------------------------------
    def n_trainable(self) -> int:
        return sum(q.numel() for q in self.parameters() if q.requires_grad)

    def n_frozen(self) -> int:
        return (sum(q.numel() for q in self.ar.parameters() if not q.requires_grad)
                + sum(b.numel() for b in self.rule.buffers()))

    # ---------------------------------------------------------------------
    def rule_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) tokens -> (B, T, len(rule_dims)) rule state, aligned to q."""
        dt = next(self.adapters.parameters()).dtype
        self._enc_logits = None

        if self.rule_input == 'tokens':
            # A one-hot is a valid distribution, so this still takes the
            # differentiable path; there is simply no encoder upstream to
            # receive the gradient.
            probs = F.one_hot(idx.long(), self.rule.vocab_size).to(dt)
        else:
            ar, T = self.ar, idx.shape[1]
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            h0  = ar.drop(ar.tok_emb(idx) + ar.pos_emb(pos))   # pre-block
            self._enc_logits = self.encoder(h0)
            # SOFTMAX, never sigmoid. The rule model's MLPs are exact on
            # inputs that sum to 1; scaling the mass 2x at one position is
            # already enough to flip a prediction, so a per-dimension
            # activation would corrupt the program. This is the same
            # constraint the chord experiment hit from the other direction.
            probs = F.softmax(self._enc_logits, dim=-1).to(dt)

        h, _ = self.rule(probs)
        return h[..., self.rule_dims]

    def encoder_loss(self, idx: torch.Tensor) -> torch.Tensor | None:
        """CE pulling the encoder's output toward the token at each position.

        Optional. Without it the encoder is shaped only by the LM loss
        reaching it through the frozen rule model and the adapter, which is
        a long and narrow path -- in the chord experiment that path alone
        left the read near chance (0.065) and the same CE term took it to
        0.340, so this is worth having even though inference never uses it.

        The target is the input token, because that is what the rule model's
        W_E would have embedded. Only position 0 actually matters to the
        program (the Select is k == 0), but supervising every position gives
        the encoder far more signal for the same forward pass.
        """
        if self._enc_logits is None:
            return None
        return F.cross_entropy(
            self._enc_logits.reshape(-1, self.rule.vocab_size),
            idx.reshape(-1).long())

    @torch.no_grad()
    def encoder_acc(self, idx: torch.Tensor) -> tuple[float, float]:
        """(all positions, position 0 only) token accuracy of the encoder.

        Position 0 is reported separately because it is the only one the
        program reads; a high overall number with a wrong seed is useless.
        """
        if self._enc_logits is None:
            return float('nan'), float('nan')
        pred = self._enc_logits.argmax(-1)
        return (float((pred == idx).float().mean()),
                float((pred[:, 0] == idx[:, 0]).float().mean()))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        ar, B, T = self.ar, idx.shape[0], idx.shape[1]
        r = self.rule_hidden(idx)

        tok = ar.tok_emb(idx)
        pos = ar.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))
        x = ar.drop(tok + pos)
        for i, blk in enumerate(ar.blocks):
            x = blk(x)
            if (i + 1) % self.n_skip == 0:
                x = x + self.adapters[(i + 1) // self.n_skip - 1](x, r)
        return ar.lm_head(ar.ln_f(x))


def build(ar_ckpt: str, rule_ckpt: str | None = None, n_skip: int = 1,
          embed_dim: int = 64, n_heads: int = 4, rule_slice: str = 'program',
          train_ar: bool = False, lora_rank: int = 0,
          lora_alpha: float | None = None, rule_input: str = 'tokens',
          encoder_hidden: int = 0, max_seq_len: int | None = None,
          device='cpu') -> IntegerRuleAdapter:
    """Load the pretrained AR and the compiled rule model, and bolt them together."""
    from models.integer_rule_model import cache_path

    blob = torch.load(ar_ckpt, map_location='cpu', weights_only=False)
    a = blob['args']
    ar = AutoregressiveTransformer(
        vocab_size=12, max_seq_len=max_seq_len or (4 * a['n_cycles'] + 8),
        d_model=a['d_model'], n_layers=a['n_layers'], n_heads=a['n_heads'])
    ar.load_state_dict(blob['state_dict'])

    rule = IntegerRuleModel.load(rule_ckpt or cache_path(32))
    m = IntegerRuleAdapter(ar, rule, n_skip=n_skip, embed_dim=embed_dim,
                           n_heads=n_heads, rule_slice=rule_slice,
                           train_ar=train_ar, lora_rank=lora_rank,
                           lora_alpha=lora_alpha, rule_input=rule_input,
                           encoder_hidden=encoder_hidden)
    return m.to(device)
