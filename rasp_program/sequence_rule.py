"""
RASP program for the modulo-4 sequence rule, compiled via tracr.

Rule:  at sequence position i the token is  (x + OFFSETS[i % 4]) % 12
       where x = tokens[0] (the starting integer) and OFFSETS = [0, 5, 7, 0].

Example (x=2): 2, 7, 9, 2, 2, 7, 9, 2, ...

Autoregressive next-token prediction
─────────────────────────────────────
At sequence position q we want to predict the token at position q+1.

Key insight (no mod-12 arithmetic needed):
  The token at cycle position c is always equal to the token at
  sequence position c in the input (for any starting integer x).
  Therefore, predicting the next token = looking up the right reference:

    predict[q] = tokens[(q + 1) % 4]

  Verification (q=0,1,2,3,4,...):
    q=0 → look at (0+1)%4 = 1 → tokens[1] = x+5   ✓
    q=1 → look at (1+1)%4 = 2 → tokens[2] = x+7   ✓
    q=2 → look at (2+1)%4 = 3 → tokens[3] = x+0=x ✓
    q=3 → look at (3+1)%4 = 0 → tokens[0] = x     ✓
    q=4 → look at (4+1)%4 = 1 → tokens[1] = x+5   ✓
    ...

This requires only ONE attention head and NO MLP operations.
Note: the TracR model requires T ≥ 4 (one full cycle) to resolve position q=2.
      FallbackRuleModel computes the result directly and works for any T ≥ 1.

Fallback
────────
If tracr / jax are not installed, FallbackRuleModel implements the same
rule directly in NumPy.  Both expose an identical API:
    .get_logits(tokens_batch: np.int32 array (B,T)) -> np.float32 array (B,T,12)
    .get_logits_vectorized(...)  (same for FallbackRuleModel; alias for TracR)
"""

from __future__ import annotations

import numpy as np
from typing import Union

VOCAB_SIZE = 12
OFFSETS    = [0, 5, 7, 0]    # period-4 rule
BOS_TOKEN  = "BOS"


# ---------------------------------------------------------------------------
# RASP program
# ---------------------------------------------------------------------------

def _build_rasp_program():
    """
    Returns a RASP s-op for next-token prediction.

    Tracr note: rasp.indices is 0-based over the *sequence* tokens;
    the prepended BOS does not appear in the index sequence.
    So seq[0]=x is at index 0, seq[1]=x+5 at index 1, seq[2]=x+7 at index 2, …

    At sequence position q, the next token is at cycle position (q+1)%3,
    which is the same as seq[(q+1)%3] (since the sequence is periodic with
    period 3 and the first cycle is always in the input).

    Therefore:  output[q] = tokens[(q+1) % 3]
    i.e., attend to key k = (q+1) % 3 for every query q.

    This requires only ONE attention head and zero MLP operations.
    """
    from tracr.rasp import rasp

    lookup_selector = rasp.Select(
        rasp.indices,   # key indices
        rasp.indices,   # query indices
        lambda k, q: k == (q + 1) % 4,
    )
    predicted_next = rasp.Aggregate(lookup_selector, rasp.tokens).named("predicted_next")
    return predicted_next


# ---------------------------------------------------------------------------
# Compiled tracr model wrapper
# ---------------------------------------------------------------------------

class TracrRuleModel:
    """
    Wraps the tracr-compiled transformer.

    get_logits(tokens_batch) -> float32 array (B, T, VOCAB_SIZE)

    tokens_batch : np.int32 array (B, T), values 0-11.
                   tokens_batch[:, 0] is the starting integer x.
    Returns one-hot logits: at position t, the one-hot of the predicted
    next token (= (x + OFFSETS[(t+1)%3]) % 12).
    """

    def __init__(self, max_seq_len: int = 128):
        self.max_seq_len = max_seq_len
        self._compiled   = None

    def _compile(self):
        from tracr.compiler import compiling

        program = _build_rasp_program()
        self._compiled = compiling.compile_rasp_to_model(
            program,
            vocab        = set(range(VOCAB_SIZE)),
            max_seq_len  = self.max_seq_len,
            compiler_bos = BOS_TOKEN,
        )

    @property
    def compiled(self):
        if self._compiled is None:
            self._compile()
        return self._compiled

    def get_logits(self, tokens_batch: np.ndarray) -> np.ndarray:
        """
        tokens_batch : int array (B, T), values in 0..11
        Returns       : float32 array (B, T, VOCAB_SIZE)  – one-hot logits
        """
        B, T = tokens_batch.shape
        out  = np.zeros((B, T, VOCAB_SIZE), dtype=np.float32)

        for b in range(B):
            seq  = [BOS_TOKEN] + tokens_batch[b].tolist()
            result = self.compiled.apply(seq)
            # decoded[0] = BOS position output (skip), decoded[1..T] = sequence outputs
            for t in range(T):
                val = result.decoded[t + 1]
                if val is not None:
                    predicted = int(round(float(val))) % VOCAB_SIZE
                    out[b, t, predicted] = 1.0
        return out

    # Alias for API compatibility with FallbackRuleModel
    def get_logits_vectorized(self, tokens_batch: np.ndarray) -> np.ndarray:
        return self.get_logits(tokens_batch)


# ---------------------------------------------------------------------------
# Pure-NumPy fallback (mathematically identical rule)
# ---------------------------------------------------------------------------

class FallbackRuleModel:
    """
    Implements the exact next-token prediction rule directly in NumPy.

    Used when tracr / jax are not installed or as a lightweight alternative.
    Mathematically equivalent to the compiled RASP/tracr model.
    """

    def get_logits(self, tokens_batch: np.ndarray) -> np.ndarray:
        return self.get_logits_vectorized(tokens_batch)

    def get_logits_vectorized(self, tokens_batch: np.ndarray) -> np.ndarray:
        """
        tokens_batch : int array (B, T), values in 0..11.
        Returns       : float32 array (B, T, VOCAB_SIZE).
                        At position t: one-hot of (x + OFFSETS[(t+1)%3]) % 12
                        where x = tokens_batch[:, 0].
        """
        B, T = tokens_batch.shape
        x    = tokens_batch[:, 0:1].astype(np.int32)   # (B, 1)
        t    = np.arange(T, dtype=np.int32)[np.newaxis, :]  # (1, T)
        offsets = np.array(OFFSETS, dtype=np.int32)
        o    = offsets[(t + 1) % 4]                         # (1, T) → broadcast
        next_tokens = (x + o) % VOCAB_SIZE                  # (B, T)

        out = np.zeros((B, T, VOCAB_SIZE), dtype=np.float32)
        rows = np.repeat(np.arange(B), T)
        cols = np.tile(np.arange(T), B)
        out[rows, cols, next_tokens.ravel()] = 1.0
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rule_model(
    max_seq_len:    int  = 128,
    force_fallback: bool = False,
) -> Union[TracrRuleModel, FallbackRuleModel]:
    """
    Returns a TracrRuleModel if tracr+jax are available, else FallbackRuleModel.
    Both expose .get_logits(tokens_batch) and .get_logits_vectorized(tokens_batch).
    """
    if force_fallback:
        print("[rule_model] Using FallbackRuleModel (forced).")
        return FallbackRuleModel()
    try:
        import tracr  # noqa: F401
        import jax    # noqa: F401
        model = TracrRuleModel(max_seq_len=max_seq_len)
        model._compile()
        print("[rule_model] Using TracrRuleModel (RASP/tracr compiled).")
        return model
    except Exception as e:
        print(f"[rule_model] tracr unavailable ({e}); using FallbackRuleModel.")
        return FallbackRuleModel()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== FallbackRuleModel ===")
    fb = FallbackRuleModel()
    # sequence: x, x+5, x+7, x,  x, x+5, x+7, x, ...  (period 4)
    # next-tok:  x+5,x+7, x, x, x+5,x+7, x,  x,  ...
    tests = [(0, [0, 5, 7, 0, 0, 5, 7, 0], [5, 7, 0, 0, 5, 7, 0, 0]),
             (2, [2, 7, 9, 2, 2, 7, 9, 2], [7, 9, 2, 2, 7, 9, 2, 2]),
             (9, [9, 2, 4, 9, 9, 2, 4, 9], [2, 4, 9, 9, 2, 4, 9, 9])]
    for x, seq, expected in tests:
        tokens = np.array([seq], dtype=np.int32)
        preds  = fb.get_logits_vectorized(tokens)[0].argmax(-1).tolist()
        ok = preds == expected
        print(f"  x={x:2d}  preds={preds}  ok={ok}")

    print("\n=== TracrRuleModel ===")
    try:
        tr = TracrRuleModel(max_seq_len=12)
        for x, seq, expected in tests:
            tokens = np.array([seq], dtype=np.int32)
            preds  = tr.get_logits(tokens)[0].argmax(-1).tolist()
            ok = preds == expected
            print(f"  x={x:2d}  preds={preds}  ok={ok}")
    except Exception as e:
        print(f"  (skipped: {e})")
