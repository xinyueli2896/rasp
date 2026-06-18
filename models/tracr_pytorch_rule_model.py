from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import VOCAB_SIZE, OFFSETS

# ─────────────────────────────────────────────────────────────────────────────
# RASP program (seed-broadcast approach):
#
#   seed_selector  = Select(indices, indices, lambda k, q: k == 0)
#   seed_broadcast = Aggregate(seed_selector, tokens)   # tokens[0] at every pos
#   predicted_next[q] = (seed_broadcast[q] + OFFSETS[(q+1)%4]) % V
#
# Why seed broadcast instead of period-4 trick?
#   The period-4 trick (attend to k=q-3) is non-causal for q<3:
#   q=0 needs q-3=-3 (invalid), q=1 needs q-2=-2 (invalid), q=2 needs q-1=-1 (invalid).
#   Seed broadcast always attends to k=0, which is causal for ALL positions.
#   tokens[0] = starter, and predicted_next[q] = (starter + OFFSETS[(q+1)%4]) % V.
#
# TRACR compilation (one attention head):
#
#   Residual stream layout  d = V*2 + 4 = 28 dims:
#     dims  0 .. V-1    : one-hot of current token t_q        (from W_E)
#     dims  V .. 2V-1   : one-hot of seed token tokens[0]     (written by attention)
#     dims 2V .. 2V+3   : one-hot of position class q%4       (from W_pos)
#
#   Seed broadcast trick:
#     W_Q maps every pos_class to pos_class 0 (constant query).
#     W_K is identity on pos subspace.
#     Score(q, k) = e_0 · e_{k%4} = 1 iff k%4==0 (k=0,4,8,...).
#     With causal mask and high attn_scale, position 0 always has highest score.
#     So all queries attend to k=0, reading tokens[0] = starter.
#
#   After residual add:
#     h_out[q] = [e_{t_q} | e_seed | e_{q%4}]  for all q >= 0
#
#   Logits via bilinear W_next:
#     W_next is a (V*4, V) lookup table where
#       W_next[(s*4 + p), :] = e_{(s + OFFSETS[(p+1)%4]) % V}
#     logits[q] = outer(e_seed, e_{q%4}) @ W_next
#
#   No trainable parameters.
# ─────────────────────────────────────────────────────────────────────────────


class TracrPyTorchRuleModel(nn.Module):
    TRACR_D_MODEL: int = VOCAB_SIZE * 2 + 4   # 28

    def __init__(self, max_seq_len: int = 128, attn_scale: float = 20.0):
        super().__init__()
        V = VOCAB_SIZE   # 12
        d = self.TRACR_D_MODEL   # 28

        self.V = V
        self.d_model = d
        self.attn_scale = attn_scale

        # W_E[t] = e_t in dims 0..V-1
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer("W_E", W_E)

        # W_pos[p] = e_p in dims 2V..2V+3
        W_pos = torch.zeros(4, d)
        for i in range(4):
            W_pos[i, 2 * V + i] = 1.0
        self.register_buffer("W_pos", W_pos)

        # W_Q: maps ALL pos_class → pos_class 0 (constant query = e_0 in pos subspace)
        #   Q[q] always has pos-class 0 → score(q,k) = 1 iff k%4==0 → attends to k=0
        W_Q = torch.zeros(d, d)
        for i in range(4):
            W_Q[2 * V, 2 * V + i] = 1.0   # all pos_class → 0
        self.register_buffer("W_Q", W_Q)

        # W_K: identity on pos subspace
        W_K = torch.zeros(d, d)
        for i in range(4):
            W_K[2 * V + i, 2 * V + i] = 1.0
        self.register_buffer("W_K", W_K)

        # W_V: copies token dims 0..V-1 into seed slot V..2V-1
        W_V = torch.zeros(d, d)
        for i in range(V):
            W_V[V + i, i] = 1.0
        self.register_buffer("W_V", W_V)

        # W_O: identity on seed slot V..2V-1
        W_O = torch.zeros(d, d)
        for i in range(V):
            W_O[V + i, V + i] = 1.0
        self.register_buffer("W_O", W_O)

        # W_next: bilinear lookup table (V*4, V)
        #   W_next[s*4 + p, :] = e_{(s + OFFSETS[(p+1)%4]) % V}
        #   Applied as: logits = outer(e_seed, e_{q%4}) @ W_next
        offsets_tensor = torch.tensor(OFFSETS, dtype=torch.long)
        seed_idx = torch.arange(V, dtype=torch.long)
        pos_idx = torch.arange(4, dtype=torch.long)
        # next_tok[s, p] = (s + OFFSETS[(p+1)%4]) % V
        next_tok = (seed_idx[:, None] + offsets_tensor[(pos_idx[None, :] + 1) % 4]) % V  # (V, 4)
        flat_idx = next_tok.reshape(-1)   # (V*4,)
        W_next = F.one_hot(flat_idx, V).float()   # (V*4, V)
        self.register_buffer("W_next", W_next)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  — token indices in [0, VOCAB_SIZE).
        Returns logits (B, T, V) and optionally the full hidden state (B, T, 28).

        logits[b, q] = predicted next-token distribution for position q.
        For all q >= 0, the rule model attends to k=0 (the seed), so
        logits[b, q] = e_{(seed + OFFSETS[(q+1)%4]) % V}.
        There is NO cold-start problem — seed is always causally available.
        """
        B, T = idx.shape
        t_idx = torch.arange(T, device=idx.device)

        # Embed: h[q] = W_E[t_q] + W_pos[q%4]
        h = self.W_E[idx] + self.W_pos[t_idx % 4].unsqueeze(0)   # (B, T, 28)

        # Queries, Keys, Values
        Q = h @ self.W_Q.t()   # all map to e_0 in pos dims 2V..2V+3
        K = h @ self.W_K.t()   # e_{q%4} in pos dims
        V = h @ self.W_V.t()   # e_{t_q} in seed slot V..2V-1

        # Causal attention scores (only pos dims contribute)
        scores = (Q @ K.transpose(-1, -2)) * self.attn_scale   # (B, T, T)
        mask = torch.ones(T, T, device=idx.device).tril().bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)                        # (B, T, T)

        # Write attended token (= seed) into seed slot, add to residual
        attn_out = (attn @ V) @ self.W_O.t()                   # (B, T, 28)
        h_out = h + attn_out                                    # (B, T, 28)

        # Logits via bilinear: outer(e_seed, e_{q%4}) @ W_next
        seed_prob = h_out[:, :, self.V : 2 * self.V]           # (B, T, V)
        pos_prob  = h_out[:, :, 2 * self.V :]                  # (B, T, 4)
        outer     = seed_prob.unsqueeze(-1) * pos_prob.unsqueeze(-2)  # (B, T, V, 4)
        outer_flat = outer.reshape(B, T, self.V * 4)
        logits    = outer_flat @ self.W_next                    # (B, T, V)

        if return_hidden:
            return logits, h_out
        return logits

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == "__main__":
    model = TracrPyTorchRuleModel()
    print(f"TRACR_D_MODEL = {model.TRACR_D_MODEL}")

    inp = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(inp, return_hidden=True)

    print(f"\nInput  : {inp.squeeze().tolist()}")
    print(f"Preds  : {logits.argmax(-1).squeeze().tolist()}")
    print(f"Expected: [5, 7, 0, 0, 5, 7, 0, 0]")

    print(f"\nHidden state breakdown (28-dim):")
    for q in range(8):
        h = hidden[0, q]
        cur  = h[:12].argmax().item()
        seed = h[12:24].argmax().item()
        pos  = h[24:].argmax().item()
        print(f"  q={q}: cur={cur}  seed={seed}  pos%4={pos}")

    tokens = inp.squeeze().tolist()
    expected = [5, 7, 0, 0, 5, 7, 0, 0]
    preds = logits.argmax(-1).squeeze().tolist()
    print(f"\nVerification (all positions, including q<3):")
    all_ok = True
    for q in range(8):
        ok = preds[q] == expected[q]
        all_ok = all_ok and ok
        tag = "" if ok else " FAIL"
        print(f"  q={q}: pred={preds[q]} (exp {expected[q]}){tag}")
    print(f"  All correct: {all_ok}")

    # Test with another starter
    print(f"\n--- Starter x=6 ---")
    from rasp_program.sequence_rule import OFFSETS
    x = 6
    seq = [(x + OFFSETS[i % 4]) % 12 for i in range(8)]
    exp = [(x + OFFSETS[(i + 1) % 4]) % 12 for i in range(8)]
    inp2 = torch.tensor([seq], dtype=torch.long)
    preds2 = model(inp2).argmax(-1).squeeze().tolist()
    print(f"  Input   : {seq}")
    print(f"  Expected: {exp}")
    print(f"  Preds   : {preds2}")
    print(f"  All OK  : {preds2 == exp}")
