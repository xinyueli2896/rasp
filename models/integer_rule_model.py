"""The integer experiment's rule model: Tracr-compiled, lifted to PyTorch.

    token[i] = (x + OFFSETS[i % 4]) % 12,   x = token[0],   OFFSETS = [0,5,7,0]

This is the frozen "Yin" stream. It emits, at every position q, a residual
state encoding the token the AR model should produce at q+1, and the adapter
cross-attends to that.

THREE PROPERTIES THE PREVIOUS RULE MODEL DID NOT HAVE
-----------------------------------------------------

1. NO FIRST CYCLE REQUIRED. `rule_hidden[0]` is already correct, so generation
   from a one-token prompt is covered. The old retrieval program was wrong at
   q = 0, 1, 2 because it looked for the answer in a cycle that had not been
   generated yet.

2. CONTEXT-INDEPENDENT. The output at q is a function of (token[0], q) and
   nothing else, which is what the rule actually says. Corrupting any token
   after the first leaves every hidden state bit-identical -- verified in
   `test_integer_rule_model.py`. The old program averaged over earlier
   positions of the matching phase, so one wrong AR token measurably degraded
   the rule signal (peak mass 1.000 -> 0.500 at q=7).

3. DIFFERENTIABLE. Input is a DISTRIBUTION over tokens, (B, T, 12), embedded
   as `probs @ W_tok` rather than looked up by index. Weights are buffers and
   `parameters()` is empty, so the model is frozen to the optimizer while
   gradients still reach whatever produced the distribution -- which is the
   whole point in the "no rule input" variant, where that distribution comes
   from a learned encoder reading the AR model's hidden states.

ALIGNMENT
---------
Tracr runs on `[BOS, t_0, ..., t_{T-1}]`. The BOS column is dropped, so

    hidden[:, q]  <->  AR position q, predicting token q+1

which is the shift-by-1 the adapter needs. The shift lives in the RASP program
(`Map(i -> (i+1) % 4)`), not in a slicing convention here, so it cannot be
silently lost by an off-by-one downstream.

WHAT ONLY POSITION 0 MATTERING IMPLIES FOR A LEARNED ENCODER
------------------------------------------------------------
With seed_mode='first' the gradient w.r.t. the input is nonzero at position 0
ONLY: the encoder has exactly one place to get it right, and no redundancy. In
the chord experiment the analogous aggregation ran over every phase-0 slot and
that vote was load-bearing -- it turned 0.31 per-position accuracy into 0.797
key accuracy. seed_mode='every_cycle' reproduces that here (`k % 4 == 0`), at
the cost of reading context again and so of being corruptible by it. Both are
exact on rule-following input; pick deliberately.

BUILD vs LOAD
-------------
Compiling needs `tracr` + `jax`; using the result does not. Build once, cache,
then load anywhere:

    python -m models.integer_rule_model --build --max_seq_len 32
    # -> checkpoints/rule_models/integer_v12_T32_first.pt

    from models.integer_rule_model import IntegerRuleModel
    rule = IntegerRuleModel.load('checkpoints/rule_models/integer_v12_T32_first.pt')
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tracr_to_torch import TracrTorch
from rasp_program.next_token_rule import (
    OFFSETS, PERIOD, VOCAB_SIZE, compile_program,
)

DEFAULT_CACHE = 'checkpoints/rule_models'


def cache_path(max_seq_len: int, vocab_size: int = VOCAB_SIZE,
               seed_mode: str = 'first', cache_dir: str = DEFAULT_CACHE) -> str:
    return os.path.join(
        cache_dir, f'integer_v{vocab_size}_T{max_seq_len}_{seed_mode}.pt')


class IntegerRuleModel(nn.Module):
    """Frozen, differentiable rule model producing shift-by-1 hidden states."""

    # Residual subspaces that are NOT computed by the program: `tokens` is a
    # verbatim echo of the input embedding, `indices` of the position, `one` is
    # Tracr's constant. They ride along in the stream but carry nothing the
    # program derived. See `program_dims`.
    ECHO_SUBSPACES = ('tokens', 'indices', 'one')

    def __init__(self, tracr_torch: TracrTorch, seed_mode: str = 'first'):
        super().__init__()
        self.net        = tracr_torch
        self.seed_mode  = seed_mode
        self.d_model    = tracr_torch.d_model
        self.vocab_size = tracr_torch.vocab_size
        self.max_seq_len = tracr_torch.max_pos - 1      # minus the BOS slot
        self.labels     = list(tracr_torch.meta['residual_labels'])
        dims = [i for i, lb in enumerate(self.labels)
                if lb.split(':')[0] not in self.ECHO_SUBSPACES]
        self.register_buffer('program_dims',
                             torch.as_tensor(dims, dtype=torch.long))

    # -- frozen ------------------------------------------------------------
    def parameters(self, recurse: bool = True):
        return iter([])

    def named_parameters(self, prefix: str = '', recurse: bool = True,
                         remove_duplicate: bool = True):
        return iter([])

    # -- named subspaces, for slicing the hidden state ---------------------
    def subspace(self, name: str) -> list[int]:
        """Residual dims belonging to one RASP variable, e.g. 'seed', 'next_tok'.

        Accepts the bare `.named()` string; Tracr's "_<id>" suffix is resolved.
        """
        exact = [i for i, lb in enumerate(self.labels) if lb.split(':')[0] == name]
        if exact:
            return exact
        hits = [i for i, lb in enumerate(self.labels)
                if lb.split(':')[0].rsplit('_', 1)[0] == name]
        if not hits:
            raise KeyError(f'no residual subspace {name!r}; have '
                           f'{sorted({lb.split(":")[0] for lb in self.labels})}')
        return hits

    # ---------------------------------------------------------------------
    def forward(self, token_probs: torch.Tensor, causal: bool = True):
        """(B, T, 12) distribution over tokens -> (hidden, logits).

        hidden : (B, T, d) -- hidden[:, q] is the rule state for predicting q+1
        logits : (B, T, 12) -- a slice of hidden, the predicted next token

        causal defaults to True here (Tracr compiles non-causal). For this
        program the two are numerically identical, because the only Select is
        `k == 0` (or `k % 4 == 0`) and those keys are always in the causal
        prefix -- the test asserts that rather than assuming it. True is the
        safe default for anything attached to an AR model.
        """
        return self.net(token_probs, prepend_bos=True, causal=causal,
                        strip_bos=True)

    def program_hidden(self, token_probs: torch.Tensor,
                       causal: bool = True) -> torch.Tensor:
        """(B, T, 12) -> (B, T, d') holding ONLY the variables the program computed.

        WHAT THIS DROPS, AND WHY IT MATTERS FOR THE ADAPTER
        ---------------------------------------------------
        Tracr's residual stream keeps the token embedding alive end to end, so
        the full stream contains `tokens` -- a verbatim one-hot of the token AT
        THAT POSITION. Injecting the full stream into the AR model therefore
        hands the adapter a copy of the current token, which (a) the AR model
        already has, and (b) is the one context-dependent channel in a model
        otherwise provably independent of context. Measured: corrupting
        token[5] moves `tokens` by 1.000 while `seed` and `next_tok` move by
        3.6e-08 and `next_phase`/`indices`/`one` by exactly 0.

        Leaving it in gives the adapter a shortcut that looks like rule-reading
        and is not, which is precisely the failure the chord probe's `local-pc`
        and `echo-base` columns exist to catch. Read this slice instead.
        """
        h, _ = self.forward(token_probs, causal=causal)
        return h[..., self.program_dims]

    def from_tokens(self, idx: torch.Tensor, causal: bool = True):
        """Convenience for hard integer input: (B, T) long -> (hidden, logits).

        No gradient path to `idx`, by nature. Use `forward` with a soft
        distribution when the input comes from a learned encoder.
        """
        probs = torch.zeros(*idx.shape, self.vocab_size,
                            device=idx.device, dtype=torch.float32)
        return self.forward(probs.scatter_(-1, idx.unsqueeze(-1).long(), 1.0),
                            causal=causal)

    # -- build / cache -----------------------------------------------------
    @staticmethod
    def build(max_seq_len: int, vocab_size: int = VOCAB_SIZE,
              offsets: list[int] | None = None,
              seed_mode: str = 'first') -> 'IntegerRuleModel':
        """Compile with Tracr and lift. Requires tracr + jax."""
        asm = compile_program(max_seq_len=max_seq_len, vocab_size=vocab_size,
                              offsets=offsets, seed_mode=seed_mode)
        net = TracrTorch.from_assembled(asm, vocab_size=vocab_size,
                                        output_name='next_tok')
        return IntegerRuleModel(net, seed_mode=seed_mode)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.net.meta['seed_mode'] = self.seed_mode
        self.net.save(path)

    @staticmethod
    def load(path: str, map_location='cpu') -> 'IntegerRuleModel':
        net = TracrTorch.load(path, map_location=map_location)
        return IntegerRuleModel(net, seed_mode=net.meta.get('seed_mode', 'first'))

    @staticmethod
    def get(max_seq_len: int, vocab_size: int = VOCAB_SIZE,
            seed_mode: str = 'first', cache_dir: str = DEFAULT_CACHE,
            ) -> 'IntegerRuleModel':
        """Load from cache, compiling and caching it first if absent."""
        p = cache_path(max_seq_len, vocab_size, seed_mode, cache_dir)
        if os.path.exists(p):
            return IntegerRuleModel.load(p)
        m = IntegerRuleModel.build(max_seq_len, vocab_size, seed_mode=seed_mode)
        m.save(p)
        return m


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--build', action='store_true',
                   help='compile with Tracr and write the cache file')
    p.add_argument('--max_seq_len', type=int, default=32,
                   help='Tracr encodes `indices` as a one-hot over absolute '
                        'positions, so d_model grows with this and the model '
                        'cannot run longer than it was compiled for.')
    p.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    p.add_argument('--seed_mode', type=str, default='first',
                   choices=['first', 'every_cycle'])
    p.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE)
    a = p.parse_args()

    if a.build:
        m = IntegerRuleModel.build(a.max_seq_len, a.vocab_size,
                                   seed_mode=a.seed_mode)
        out = cache_path(a.max_seq_len, a.vocab_size, a.seed_mode, a.cache_dir)
        m.save(out)
        print(f'compiled and cached -> {out}')
    else:
        m = IntegerRuleModel.get(a.max_seq_len, a.vocab_size, a.seed_mode,
                                 a.cache_dir)

    print(f'rule: token[i] = (x + {OFFSETS}[i % {PERIOD}]) % {m.vocab_size}')
    print(f'd_model={m.d_model}  max_seq_len={m.max_seq_len}  '
          f'seed_mode={m.seed_mode}  trainable={len(list(m.parameters()))}')
    names = sorted({lb.split(':')[0] for lb in m.labels})
    print('residual subspaces: ' + ', '.join(
        f'{n}[{len(m.subspace(n))}]' for n in names))

    idx = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]])
    h, lg = m.from_tokens(idx)
    print(f'\ninput        : {idx[0].tolist()}')
    print(f'predicts next: {lg.argmax(-1)[0].tolist()}')
    print(f'expected     : {[(0 + OFFSETS[(q+1) % 4]) % m.vocab_size for q in range(8)]}')
    print(f'hidden       : {tuple(h.shape)}  (hidden[:, q] -> AR position q)')


if __name__ == '__main__':
    _main()
