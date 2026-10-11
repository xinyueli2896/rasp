"""RASP program for next-token prediction under the modulo rule.

    token[i] = (x + OFFSETS[i % 4]) % V,    x = token[0]

WHAT CHANGED, AND WHY
---------------------
The previous program (`sequence_rule.py`) predicted by RETRIEVAL:

    predicted_next[q] = tokens[(q + 1) % 4]

That exploits the fact that a periodic sequence repeats its first cycle, so
the answer is always sitting somewhere earlier in the context. It needs no
arithmetic -- one attention head, no MLP -- which is elegant, but it buys that
elegance with two defects that matter once the model is attached to an
autoregressive transformer:

  1. IT NEEDS A FIRST CYCLE IN CONTEXT. At q = 0, 1, 2 no causal key has the
     required position class, so attention falls back to a uniform average
     over the prefix and the output is wrong. Generation starts from a
     ONE-TOKEN prompt, so those are precisely the positions that set up
     everything that follows.

  2. IT READS THE CONTEXT, SO IT INHERITS THE CONTEXT'S MISTAKES. The
     aggregate averages over every earlier position of the matching class.
     On rule-following input those positions all carry the same token and the
     average is lossless; on input the AR model got wrong they disagree and
     the rule signal smears. Measured on the old model: corrupting one token
     at t=4 drops the peak output mass at q=7 from 1.000 to 0.500.

The rule does not actually depend on the context. It depends on exactly two
things: the FIRST token, and WHICH POSITION is being predicted. So compute it:

    seed      = Aggregate(Select(indices, indices, k == 0), tokens)
    nextphase = Map(i -> (i + 1) % 4, indices)
    next_tok  = SequenceMap((s, p) -> (s + OFFSETS[p]) % V, seed, nextphase)

Position 0 is causally available to every query, so this is exact from T = 1,
and nothing but token 0 can influence the answer -- corrupting any later token
leaves the output bit-identical.

THE COST: this needs real arithmetic, so the SequenceMap compiles to an MLP
(Tracr compiles every elementwise/pairwise map to an MLP and only Select/
Aggregate to attention). The model is 2 layers instead of 1. That is the same
head-plus-MLP shape as the chord experiment's ChordRaspCompiled, which is the
point: retrieval alone cannot express this rule, and pretending otherwise is
what forced the first-cycle assumption.

SHIFT-BY-1 is built into the program rather than bolted on afterwards: the
`(i + 1) % 4` means position q emits the token for position q+1, which is what
an AR model at position q is trying to predict.

`seed_mode`
-----------
  'first'       k == 0.       The seed is read from position 0 only.
  'every_cycle' k % 4 == 0.   The seed is averaged over every cycle start.

Both are exact on rule-following input ('every_cycle' averages one-hots that
agree). They differ off-rule, and in how much leverage a LEARNED encoder has:
with 'first' a single position carries the whole signal, with 'every_cycle'
T/4 positions vote. In the chord experiment that vote was load-bearing -- it
turned 0.31 per-position root accuracy into 0.797 key accuracy -- so it is
exposed here rather than hard-coded.
"""
from __future__ import annotations

import numpy as np

VOCAB_SIZE = 12
OFFSETS    = [0, 5, 7, 0]
PERIOD     = len(OFFSETS)
BOS_TOKEN  = "BOS"


def build_program(vocab_size: int = VOCAB_SIZE,
                  offsets: list[int] | None = None,
                  seed_mode: str = 'first'):
    """The RASP s-op. Requires `tracr` to be installed."""
    from tracr.rasp import rasp

    offsets = list(offsets if offsets is not None else OFFSETS)
    period  = len(offsets)
    assert seed_mode in ('first', 'every_cycle'), seed_mode

    if seed_mode == 'first':
        pred = lambda k, q: k == 0
    else:
        pred = lambda k, q: k % period == 0

    seed_sel = rasp.Select(rasp.indices, rasp.indices, pred)
    seed     = rasp.Aggregate(seed_sel, rasp.tokens).named("seed")
    nphase   = rasp.Map(lambda i: (i + 1) % period, rasp.indices).named("next_phase")
    return rasp.SequenceMap(
        lambda s, p: (s + offsets[p]) % vocab_size, seed, nphase).named("next_tok")


def compile_program(max_seq_len: int,
                    vocab_size: int = VOCAB_SIZE,
                    offsets: list[int] | None = None,
                    seed_mode: str = 'first'):
    """Compile to a Tracr AssembledTransformerModel. Requires `tracr`."""
    from tracr.compiler import compiling

    return compiling.compile_rasp_to_model(
        build_program(vocab_size, offsets, seed_mode),
        vocab        = set(range(vocab_size)),
        max_seq_len  = max_seq_len,
        compiler_bos = BOS_TOKEN,
    )


# ---------------------------------------------------------------------------
# NumPy oracle — the ground truth every other implementation is checked against
# ---------------------------------------------------------------------------

def next_token(starters, seq_len: int,
               vocab_size: int = VOCAB_SIZE,
               offsets: list[int] | None = None) -> np.ndarray:
    """(B,) starters -> (B, T) the token each position PREDICTS (position q+1)."""
    offs = np.asarray(offsets if offsets is not None else OFFSETS)
    x    = np.asarray(starters, dtype=np.int64).reshape(-1, 1)
    q    = np.arange(seq_len, dtype=np.int64)[None, :]
    return (x + offs[(q + 1) % len(offs)]) % vocab_size


def sequence(starters, seq_len: int,
             vocab_size: int = VOCAB_SIZE,
             offsets: list[int] | None = None) -> np.ndarray:
    """(B,) starters -> (B, T) the rule-following sequence itself."""
    offs = np.asarray(offsets if offsets is not None else OFFSETS)
    x    = np.asarray(starters, dtype=np.int64).reshape(-1, 1)
    q    = np.arange(seq_len, dtype=np.int64)[None, :]
    return (x + offs[q % len(offs)]) % vocab_size
