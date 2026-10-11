"""Convert a Tracr-compiled transformer into a frozen, DIFFERENTIABLE PyTorch module.

WHY A CONVERTER AND NOT HAND-WRITTEN WEIGHTS
--------------------------------------------
The previous rule models (`tracr_pytorch_rule_model.py`, `bass_tracr_rule_model.py`,
`chord_tracr_rule_model.py`) are hand-constructed transformers "equivalent to"
what Tracr would emit. That equivalence is an unverified claim in a docstring,
and at least one of those docstrings was wrong (it asserted sharp attention to
k=q-3; the real pattern averages over every causal key of the matching position
class). Here we compile the RASP program with the real Tracr and LIFT the
resulting weights, so there is nothing to get wrong by hand -- and
`verify_against_tracr` re-checks the lift numerically.

WHY NOT JUST USE TRACR AT TRAINING TIME
---------------------------------------
Two reasons:

  1. GRADIENT FLOW. The rule model sits inside a PyTorch training graph. Its
     weights are frozen, but gradients must pass THROUGH it to whatever
     produces its input -- the learned encoder in the "no rule input" variant.
     A JAX model cannot be in that graph.

  2. SOFT INPUTS. Tracr embeds via `hk.Embed(tokens)`, an integer lookup, which
     has no gradient with respect to the input at all. This module takes a
     DISTRIBUTION over tokens, (B, T, V), and embeds as `probs @ W_tok`. On a
     hard one-hot that is exactly the lookup; on a soft input it is the
     differentiable relaxation the encoder needs.

Everything is registered with `register_buffer`, and `parameters()` is
overridden to return nothing, so no optimizer can ever touch the weights while
autograd still flows through them.

FIDELITY
--------
Replicates `tracr/transformer/model.py::Transformer.__call__` exactly:

  * embed      h = W_tok[tok] + W_pos[position]   (position 0 is the BOS slot,
               and `pos_embed` row 0 is all zeros by construction)
  * attention  logits = q k^T / sqrt(key_size); masked with -1e30, not -inf
  * mask       applies to KEY positions, broadcast across queries
  * NO LAYER NORM when the compiled config says layer_norm=False (it does;
    LayerNorm would mix Tracr's disjoint subspaces and destroy the one-hots)
  * mlp        linear_1 -> relu -> linear_2, residual-added
  * output     the residual stream itself; "logits" are a SLICE of it

Usage
-----
    from rasp_program.next_token_rule import compile_program
    from models.tracr_to_torch import TracrTorch

    assembled = compile_program(max_seq_len=32)        # needs tracr + jax
    torch_model = TracrTorch.from_assembled(assembled)
    torch_model.save('checkpoints/rule_int_v12_T32.pt')

    # later, on a machine with no jax:
    torch_model = TracrTorch.load('checkpoints/rule_int_v12_T32.pt')
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tracr masks with a large negative constant rather than -inf. Reproduced
# exactly: -inf would make a fully-masked row NaN instead of uniform, and the
# two differ in the last bits elsewhere.
TRACR_MASK_VALUE = -1e30


class TracrTorch(nn.Module):
    """A Tracr-compiled transformer, in PyTorch, frozen but differentiable."""

    def __init__(self, cfg: dict, weights: dict, meta: dict):
        super().__init__()
        self.cfg  = dict(cfg)
        self.meta = dict(meta)
        self.n_layers  = int(cfg['num_layers'])
        self.n_heads   = int(cfg['num_heads'])
        self.key_size  = int(cfg['key_size'])
        self.layer_norm = bool(cfg['layer_norm'])
        self.causal_default = bool(cfg['causal'])

        if self.layer_norm:
            raise NotImplementedError(
                'This converter assumes the compiled config has layer_norm=False, '
                'which is what compile_rasp_to_model produces. A LayerNorm would '
                'mix the disjoint residual subspaces and the lift would not be '
                'faithful without also carrying its scale/offset.')

        for name, w in weights.items():
            t = (w.detach().clone() if torch.is_tensor(w)
                 else torch.tensor(w, dtype=torch.float32))
            self.register_buffer(name.replace('/', '__').replace('.', '_'),
                                 t.to(torch.float32))
        self._wnames = {n: n.replace('/', '__').replace('.', '_') for n in weights}

        self.d_model   = self.W('token_embed__embeddings').shape[1]
        self.n_tokens  = self.W('token_embed__embeddings').shape[0]
        self.max_pos   = self.W('pos_embed__embeddings').shape[0]
        self.vocab_size = int(meta['vocab_size'])
        self.bos_id     = int(meta['bos_id'])
        self.pad_id     = meta.get('pad_id', None)
        # Dims of the residual stream that hold the program's output variable,
        # in output-encoder order. Tracr's unembed is a projection onto these.
        self.register_buffer('out_dims',
                             torch.as_tensor(meta['out_dims'], dtype=torch.long))

    # -- buffer access -----------------------------------------------------
    def W(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace('/', '__').replace('.', '_'))

    def _p(self, module: str, kind: str) -> torch.Tensor:
        return self.W(f'{module}__{kind}')

    # -- no trainable parameters, ever -------------------------------------
    def parameters(self, recurse: bool = True):
        return iter([])

    def named_parameters(self, prefix: str = '', recurse: bool = True,
                         remove_duplicate: bool = True):
        return iter([])

    # ---------------------------------------------------------------------
    def embed(self, token_probs: torch.Tensor,
              prepend_bos: bool = True) -> torch.Tensor:
        """(B, T, V) distribution over tokens -> (B, T[+1], d) residual stream.

        `probs @ W_tok[:V]` is the differentiable form of Tracr's integer
        `hk.Embed`; on a one-hot the two are identical.
        """
        B, T, V = token_probs.shape
        if V != self.vocab_size:
            raise ValueError(f'expected last dim {self.vocab_size}, got {V}')
        W_tok = self.W('token_embed__embeddings')
        W_pos = self.W('pos_embed__embeddings')
        # The lifted weights are float32. Cast rather than letting the matmul
        # raise "expected m1 and m2 to have the same dtype", which says nothing
        # about which argument is wrong. Casting is lossless in the direction
        # that matters: a float64 caller loses nothing the float32 weights
        # could have used.
        if token_probs.dtype != W_tok.dtype:
            token_probs = token_probs.to(W_tok.dtype)

        h = token_probs @ W_tok[:self.vocab_size]            # (B, T, d)
        if prepend_bos:
            bos = W_tok[self.bos_id].expand(B, 1, -1)
            h   = torch.cat([bos, h], dim=1)                 # (B, T+1, d)
        T_tot = h.shape[1]
        if T_tot > self.max_pos:
            raise ValueError(
                f'sequence of {T_tot} (incl. BOS) exceeds the max_seq_len this '
                f'program was compiled for ({self.max_pos - 1}). Tracr encodes '
                f'`indices` as a one-hot over absolute positions, so the model '
                f'must be recompiled to go longer.')
        return h + W_pos[:T_tot].unsqueeze(0)

    def _attention(self, h: torch.Tensor, layer: int,
                   key_mask: torch.Tensor, causal: bool) -> torch.Tensor:
        B, T, _ = h.shape
        H, ks = self.n_heads, self.key_size
        pre = f'transformer__layer_{layer}__attn'

        def proj(kind):
            x = h @ self._p(f'{pre}__{kind}', 'w') + self._p(f'{pre}__{kind}', 'b')
            return x.view(B, T, H, ks).transpose(1, 2)       # (B, H, T, ks)

        q, k, v = proj('query'), proj('key'), proj('value')
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(ks)   # (B, H, T, T)

        # Tracr masks KEY positions and broadcasts across queries.
        m = key_mask[:, None, None, :].expand(B, H, T, T)
        if causal:
            m = m & torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        logits = torch.where(m, logits, logits.new_full((), TRACR_MASK_VALUE))

        a   = F.softmax(logits, dim=-1)
        out = (a @ v).transpose(1, 2).reshape(B, T, H * ks)
        return out @ self._p(f'{pre}__linear', 'w') + self._p(f'{pre}__linear', 'b')

    def _mlp(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        pre = f'transformer__layer_{layer}__mlp'
        x = F.relu(h @ self._p(f'{pre}__linear_1', 'w')
                   + self._p(f'{pre}__linear_1', 'b'))
        return x @ self._p(f'{pre}__linear_2', 'w') + self._p(f'{pre}__linear_2', 'b')

    def forward(self, token_probs: torch.Tensor,
                prepend_bos: bool = True,
                causal: bool | None = None,
                strip_bos: bool = True):
        """(B, T, V) -> (hidden, logits).

        hidden : (B, T, d) residual stream, BOS column dropped when strip_bos.
        logits : (B, T, V) the program's output variable, a slice of `hidden`.

        causal=None uses the compiled config (False). The only Select in the
        integer program is on key position alone, so a causal mask is a no-op
        there -- but pass causal=True if you want that guaranteed structurally
        rather than argued from the program text.
        """
        causal = self.causal_default if causal is None else causal
        h = self.embed(token_probs, prepend_bos=prepend_bos)
        B, T, _ = h.shape
        key_mask = torch.ones(B, T, dtype=torch.bool, device=h.device)

        for i in range(self.n_layers):
            h = h + self._attention(h, i, key_mask, causal)
            h = h + self._mlp(h, i)

        if strip_bos and prepend_bos:
            h = h[:, 1:]
        return h, h[..., self.out_dims]

    # ---------------------------------------------------------------------
    # Construction from a Tracr AssembledTransformerModel
    # ---------------------------------------------------------------------
    @staticmethod
    def from_assembled(assembled, vocab_size: int,
                       output_name: str | None = None) -> 'TracrTorch':
        """Lift `assembled.params` into buffers. Requires jax only at this step."""
        import numpy as np

        mc = assembled.model_config
        cfg = dict(num_layers=mc.num_layers, num_heads=mc.num_heads,
                   key_size=mc.key_size, mlp_hidden_size=mc.mlp_hidden_size,
                   layer_norm=mc.layer_norm, causal=mc.causal)

        weights = {}
        for mod, inner in assembled.params.items():
            for kind, arr in inner.items():
                weights[f'{mod}/{kind}'] = np.asarray(arr, dtype=np.float32)

        labels = list(assembled.residual_labels)
        # Dims of the output s-op, in the order the output encoder decodes them.
        # Tracr renames each s-op to "<name>_<id>", so an intermediate variable
        # can have the same width and value set as the output (here `seed_2`
        # and `next_tok_1` are both 12-wide over 0..11). Resolve by NAME --
        # guessing by shape would silently read the wrong subspace.
        cands = sorted({lb.split(':')[0] for lb in labels
                        if ':' in lb and lb.split(':')[0]
                        not in ('indices', 'tokens', 'one')})
        if output_name is None:
            if len(cands) != 1:
                raise ValueError(
                    f'cannot infer the output subspace among {cands}; pass '
                    f'output_name= (the s-op name you gave .named(), with or '
                    f'without Tracr\'s "_<id>" suffix)')
            output_name = cands[0]
        elif output_name not in cands:
            # accept the bare .named() string, e.g. "next_tok" -> "next_tok_1"
            hits = [c for c in cands if c.rsplit('_', 1)[0] == output_name]
            if len(hits) != 1:
                raise ValueError(
                    f'output_name={output_name!r} matched {hits} among {cands}')
            output_name = hits[0]
        oe = assembled.output_encoder
        inv = {v: k for k, v in oe.encoding_map.items()}
        out_dims = []
        for i in range(len(inv)):
            lab = f'{output_name}:{inv[i]}'
            if lab not in labels:
                raise ValueError(f'residual label {lab!r} not found in {labels}')
            out_dims.append(labels.index(lab))

        ie = assembled.input_encoder
        meta = dict(vocab_size=vocab_size,
                    bos_id=int(ie.encoding_map[ie.bos_token]),
                    pad_id=int(ie.encoding_map[ie.pad_token])
                    if getattr(ie, 'pad_token', None) in ie.encoding_map else None,
                    out_dims=out_dims,
                    output_name=output_name,
                    residual_labels=labels)
        return TracrTorch(cfg, weights, meta)

    # ---------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({'cfg': self.cfg, 'meta': self.meta,
                    'weights': {n: self.W(n).cpu() for n in self._wnames}}, path)

    @staticmethod
    def load(path: str, map_location='cpu') -> 'TracrTorch':
        blob = torch.load(path, map_location=map_location, weights_only=False)
        return TracrTorch(blob['cfg'], blob['weights'], blob['meta'])


# ---------------------------------------------------------------------------
def verify_against_tracr(torch_model: 'TracrTorch', assembled,
                         sequences, bos_token="BOS", atol=1e-5):
    """Residual-stream agreement between the lift and the real Tracr model.

    Checks the FULL residual stream, not just the decoded output: two models
    can decode identically while disagreeing everywhere the adapter will
    actually read.
    """
    import numpy as np

    worst, rows = 0.0, []
    for seq in sequences:
        enc = assembled.input_encoder.encode([bos_token] + list(seq))
        jx  = assembled.forward(assembled.params, np.array([enc]))
        ref = np.asarray(jx.transformer_output.residuals[-1])[0]      # (T+1, d)

        probs = torch.zeros(1, len(seq), torch_model.vocab_size)
        probs[0, torch.arange(len(seq)), torch.as_tensor(seq)] = 1.0
        h, _ = torch_model(probs, strip_bos=False)
        got = h[0].detach().numpy()

        d = float(np.abs(got - ref).max())
        worst = max(worst, d)
        rows.append((list(seq), d))
    return worst, rows
