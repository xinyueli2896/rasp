"""
Encoder for the no-rule-input variant.

The rule model is normally handed the token sequence.  Here it is handed
nothing: the AR model's own representation BEFORE the transformer blocks,

    x0 = drop(tok_emb(idx) + pos_emb(pos))        # (B, T, d_model)

is passed through a learned encoder whose output replaces the rule model's
input embedding.  Nothing external reaches the rule model at inference, so the
rule signal has to be recovered from the host model's own activations.

    x0  --encoder-->  p (B, T, 12)  -->  rule model's token subspace
                                          |
                                   frozen head: retrieve the key
                                   frozen MLP:  (key, phase) -> next root

Why softmax, and not a free linear output
-----------------------------------------
The compiled MLP gates each (key, phase) pair with ReLU(key_k + phase_p - 1),
which assumes the key subspace carries at most 1.0 in any dimension.  Feed it
a token vector scaled by 2 and the gate fires on the key alone, with no phase:
the root subspace fills with (key + OFFSETS[p]) % 12 for EVERY p at once.
Measured on key 6, phase 1: the correct {6: 1.00} becomes
{1: 1.00, 6: 3.00, 11: 1.00}, and the argmax is wrong wherever the spurious
roots outweigh the real one.  Raw logits fail the same way, and worse, since
negative values subtract from the gate.

softmax guarantees every dimension lies in [0, 1] and the row sums to 1, which
is exactly the precondition the gate needs.  A soft (low-confidence) output
still decodes correctly -- it only shrinks the amplitude of the root signal the
adapter receives, which is the honest behaviour: an unsure encoder produces a
weak rule signal rather than a confident wrong one.

Causality
---------
x0[t] = tok_emb(idx[t]) + pos_emb(t) depends on token t and position t only,
never on t+1, so reading it introduces no leak.  The rule model's own head is
causal on top of that.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ROOTS = 12


class ARStateRuleEncoder(nn.Module):
    """(B, T, d_model) AR pre-block states -> (B, T, 12) distribution.

    encoder_type:
      'linear' : Linear(d_model -> 12)
      'mlp'    : Linear(d_model -> hidden) -> ReLU -> Linear(hidden -> 12)
    """

    def __init__(self, d_model: int, vocab_size: int = N_ROOTS,
                 encoder_type: str = 'mlp', hidden: int = 64):
        super().__init__()
        if encoder_type not in ('linear', 'mlp'):
            raise ValueError(
                f"encoder_type must be 'linear' or 'mlp', got {encoder_type!r}")
        self.encoder_type = encoder_type
        self.vocab_size   = vocab_size

        if encoder_type == 'linear':
            self.fc_out = nn.Linear(d_model, vocab_size)
            nn.init.xavier_uniform_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)
        else:
            self.fc1    = nn.Linear(d_model, hidden)
            self.fc_out = nn.Linear(hidden, vocab_size)
            for lin in (self.fc1, self.fc_out):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)

    def logits(self, x0: torch.Tensor) -> torch.Tensor:
        """Pre-softmax scores (B, T, 12) -- what the auxiliary CE supervises."""
        if self.encoder_type == 'linear':
            return self.fc_out(x0)
        return self.fc_out(F.relu(self.fc1(x0)))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """(B, T, 12) distribution, safe to write into the rule model."""
        return F.softmax(self.logits(x0), dim=-1)
