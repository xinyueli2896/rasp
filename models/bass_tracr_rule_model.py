"""
Analytical TracR-style rule model for bass pitch-class prediction.

Exact structural analogue of TracrPyTorchRuleModel (integer model, mod 24),
but operating on pitch classes (mod 12) rather than integers (mod 24).

Rule:  sequence[t] = (x + OFFSETS[t % 4]) % 12,   OFFSETS = [0, 5, 7, 0]
       where x = pitch_classes[:, 0]  (starting pitch class, read from sequence)

Input : pitch class indices  (B, T)   int64, values in 0-11
Output: logits               (B, T, 12)
        hidden               (B, T, d_model)   if return_hidden=True

rule_mode='current' (default, d_model=16):
  Hidden state format:
    dims  0-11 : one-hot of CURRENT pitch class at position t  (key + OFFSETS[t%4])
    dims 12-15 : one-hot of t % 4
  Analytical forward (no attention). TRACR_D_MODEL class attribute = 16.

rule_mode='period4' (d_model=28):
  Uses TRACR attention (causal mask).
  dims  0-11 : one-hot of CURRENT pitch class at position t
  dims 12-23 : one-hot of NEXT pitch class  (key + OFFSETS[(t+1)%4])
  dims 24-27 : one-hot of t % 4
  W_Q shifts pos_class +1: W_Q[24+(i+1)%4, 24+i] = 1

rule_mode='seed_broadcast' (d_model=28):
  Uses TRACR attention (causal mask).
  dims  0-11 : one-hot of CURRENT pitch class at position t
  dims 12-23 : one-hot of SEED pitch class  (key = pc_0, constant)
  dims 24-27 : one-hot of t % 4
  W_Q maps all pos_class→0: W_Q[24+0, 24+i] = 1 for all i

No trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import OFFSETS

N_ROOTS  = 12
N_POS    = 4
TRACR_D_MODEL = N_ROOTS + N_POS   # 16  — kept for backward compat


class BassTracrRuleModel(nn.Module):

    TRACR_D_MODEL: int = TRACR_D_MODEL   # class attribute, always 16

    def __init__(
        self,
        rule_mode:    str   = 'current',
        max_seq_len:  int   = 512,
        attn_scale:   float = 20.0,
    ):
        assert rule_mode in ('current', 'period4', 'seed_broadcast'), \
            f"rule_mode must be 'current', 'period4', or 'seed_broadcast', got {rule_mode!r}"
        super().__init__()
        self.rule_mode  = rule_mode
        self.attn_scale = attn_scale

        V = N_ROOTS

        if rule_mode == 'current':
            d = N_ROOTS + N_POS   # 16
        else:
            d = N_ROOTS * 2 + N_POS   # 28

        # Instance attribute — reflects the actual dimension for this instance
        self.d_model = d

        # W_E[r] = one-hot(r) in dims 0..V-1, zeros elsewhere
        W_E = torch.zeros(V, d)
        W_E[:V, :V] = torch.eye(V)
        self.register_buffer('W_E', W_E)

        # W_pos[p] = one-hot(p) in dims (d-N_POS)..(d-1)
        W_pos = torch.zeros(N_POS, d)
        for i in range(N_POS):
            W_pos[i, d - N_POS + i] = 1.0
        self.register_buffer('W_pos', W_pos)

        if rule_mode != 'current':
            # --- attention matrices for period4 / seed_broadcast ---
            pos_base = d - N_POS   # = 24

            # W_Q  (d, d)
            W_Q = torch.zeros(d, d)
            if rule_mode == 'period4':
                # shift pos_class +1: query pos i ← key pos (i+1)%4
                for i in range(N_POS):
                    W_Q[pos_base + (i + 1) % N_POS, pos_base + i] = 1.0
            else:  # seed_broadcast: all pos → slot 0
                for i in range(N_POS):
                    W_Q[pos_base + 0, pos_base + i] = 1.0
            self.register_buffer('W_Q', W_Q)

            # W_K: identity on pos subspace (dims pos_base..d-1)
            W_K = torch.zeros(d, d)
            for i in range(N_POS):
                W_K[pos_base + i, pos_base + i] = 1.0
            self.register_buffer('W_K', W_K)

            # W_V: copies dims 0-11 to dims 12-23
            W_V = torch.zeros(d, d)
            for i in range(V):
                W_V[V + i, i] = 1.0
            self.register_buffer('W_V', W_V)

            # W_O: identity on dims 12-23
            W_O = torch.zeros(d, d)
            for i in range(V):
                W_O[V + i, V + i] = 1.0
            self.register_buffer('W_O', W_O)

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  pitch class indices in 0-11; idx[:, 0] determines the key.
        """
        B, T   = idx.shape
        device = idx.device
        t_idx  = torch.arange(T, device=device)

        key   = idx[:, 0]                                    # (B,)
        o_cur = self._offsets[t_idx % N_POS]                 # (T,) CURRENT offsets
        cur_r = (key[:, None] + o_cur[None, :]) % N_ROOTS   # (B, T)

        if self.rule_mode == 'current':
            logits = F.one_hot(cur_r, num_classes=N_ROOTS).float()
            if return_hidden:
                tok_cur = self.W_E[cur_r]                    # (B, T, 16)
                pos_emb = self.W_pos[t_idx % N_POS]         # (T, 16)
                h_out   = tok_cur + pos_emb.unsqueeze(0)    # (B, T, 16)
                return logits, h_out
            return logits

        # --- period4 / seed_broadcast: run TRACR attention ---
        # Build residual stream embeddings
        tok_cur = self.W_E[cur_r]                            # (B, T, 28) — dims 0-11
        pos_emb = self.W_pos[t_idx % N_POS]                 # (T, 28) — dims 24-27
        x = tok_cur + pos_emb.unsqueeze(0)                  # (B, T, 28)

        # Queries, keys, values
        Q = x @ self.W_Q.T                                  # (B, T, 28)
        K = x @ self.W_K.T                                  # (B, T, 28)
        V = x @ self.W_V.T                                  # (B, T, 28)

        # Attention scores with causal mask
        scores = (Q @ K.transpose(-2, -1)) * self.attn_scale  # (B, T, T)
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        attn   = F.softmax(scores, dim=-1)                   # (B, T, T)

        # Weighted sum, project through W_O, add residual
        h_out = x + (attn @ V) @ self.W_O.T                 # (B, T, 28)

        # Logits from dims 12-23 (next/seed pc)
        logits = h_out[:, :, N_ROOTS:N_ROOTS * 2]           # (B, T, 12)

        if return_hidden:
            return logits, h_out
        return logits

    # ------------------------------------------------------------------
    # Analytic rule_hidden construction (no attention needed)
    # ------------------------------------------------------------------

    def build_rule_hidden_analytic(
        self,
        pc_0:   torch.Tensor,   # (B,) starting pitch class
        T:      int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build rule_hidden analytically without running attention.

        Returns (B, T, d_model).

        current   : dims 0-11 = one_hot((pc_0+OFFSETS[t%4])%12),  dims 12-15 = one_hot(t%4)
        period4   : dims 0-11 = current pc, dims 12-23 = next pc,  dims 24-27 = one_hot(t%4)
        seed_bcast: dims 0-11 = current pc, dims 12-23 = seed pc,  dims 24-27 = one_hot(t%4)
        """
        B      = pc_0.shape[0]
        t_idx  = torch.arange(T, device=device)
        offsets_buf = self._offsets.to(device)

        o_cur  = offsets_buf[t_idx % N_POS]                 # (T,)
        cur_r  = (pc_0[:, None].to(device) + o_cur[None, :]) % N_ROOTS  # (B, T)

        d = self.d_model
        h = torch.zeros(B, T, d, device=device)

        # dims 0-11: current pitch class
        h[:, :, :N_ROOTS] = F.one_hot(cur_r, num_classes=N_ROOTS).float()

        if self.rule_mode == 'current':
            # dims 12-15: pos_class
            pos_oh = F.one_hot(t_idx % N_POS, num_classes=N_POS).float()  # (T, 4)
            h[:, :, N_ROOTS:N_ROOTS + N_POS] = pos_oh.unsqueeze(0).expand(B, -1, -1)

        elif self.rule_mode == 'period4':
            # dims 12-23: next pitch class
            o_next  = offsets_buf[(t_idx + 1) % N_POS]      # (T,)
            next_r  = (pc_0[:, None].to(device) + o_next[None, :]) % N_ROOTS  # (B, T)
            h[:, :, N_ROOTS:N_ROOTS * 2] = F.one_hot(next_r, num_classes=N_ROOTS).float()
            # dims 24-27: pos_class
            pos_oh = F.one_hot(t_idx % N_POS, num_classes=N_POS).float()
            h[:, :, N_ROOTS * 2:] = pos_oh.unsqueeze(0).expand(B, -1, -1)

        else:  # seed_broadcast
            # dims 12-23: seed/key pitch class (constant)
            seed_oh = F.one_hot(pc_0.to(device) % N_ROOTS, num_classes=N_ROOTS).float()  # (B, 12)
            h[:, :, N_ROOTS:N_ROOTS * 2] = seed_oh.unsqueeze(1).expand(-1, T, -1)
            # dims 24-27: pos_class
            pos_oh = F.one_hot(t_idx % N_POS, num_classes=N_POS).float()
            h[:, :, N_ROOTS * 2:] = pos_oh.unsqueeze(0).expand(B, -1, -1)

        return h

    # ------------------------------------------------------------------

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


class CPChordRuleModel(nn.Module):
    """
    Rule model for chord-progression regulation (bar-level I-IV-V-I).

    One chord per bar: the I-IV-V-I cadence spans 4 bars, not 4 beats.

        bar   = t // beats_per_bar
        root  = (key + OFFSETS[bar % 4]) % 12
        chord = {root, root+4, root+7}  (major triad)

    Outputs a 12-dim binary chromagram of the expected major triad at each beat.

    Input : pitch-class sequence  (B, T)   int64, idx[:, 0] = key
    Output: chromagram targets    (B, T, 12) float binary
            hidden                (B, T, d_model=16) if return_hidden=True

    Hidden state format (d_model=16):
        dims  0-11 : 12-dim binary chromagram of expected major triad
        dims 12-15 : one-hot of bar % 4  (bar-phase positional class)

    No trainable parameters.
    """

    CHORD_INTERVALS: tuple = (0, 4, 7)   # major triad
    d_model:         int   = N_ROOTS + N_POS   # 16
    TRACR_D_MODEL:   int   = N_ROOTS + N_POS   # 16

    def __init__(self, beats_per_bar: int = 4):
        super().__init__()
        self.beats_per_bar = beats_per_bar
        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))
        W_pos = torch.zeros(N_POS, self.d_model)
        for i in range(N_POS):
            W_pos[i, N_ROOTS + i] = 1.0
        self.register_buffer('W_pos', W_pos)

    # ------------------------------------------------------------------

    def _bar_phase(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Bar-phase index: which slot in OFFSETS does beat t belong to."""
        return (t_idx // self.beats_per_bar) % N_POS

    def _root_to_chromagram(self, root: torch.Tensor) -> torch.Tensor:
        """root: (B, T)  →  chromagram: (B, T, 12) binary float."""
        chroma = torch.zeros(*root.shape, N_ROOTS,
                             device=root.device, dtype=torch.float)
        for interval in self.CHORD_INTERVALS:
            idx = (root + interval) % N_ROOTS
            chroma.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return chroma

    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        """
        idx : (B, T)  pitch-class sequence; idx[:, 0] is the key.
        Returns chromagram (B, T, 12) float binary, optionally also hidden (B, T, 16).
        """
        B, T   = idx.shape
        device = idx.device
        t_idx  = torch.arange(T, device=device)

        key      = idx[:, 0]                                          # (B,)
        phase    = self._bar_phase(t_idx)                             # (T,)
        o_cur    = self._offsets[phase]                               # (T,)
        cur_root = (key[:, None] + o_cur[None, :]) % N_ROOTS         # (B, T)
        chroma   = self._root_to_chromagram(cur_root)                 # (B, T, 12)

        if return_hidden:
            h = torch.zeros(B, T, self.d_model, device=device)
            h[:, :, :N_ROOTS] = chroma
            pos_oh = F.one_hot(phase, num_classes=N_POS).float()      # (T, 4)
            h[:, :, N_ROOTS:] = pos_oh.unsqueeze(0).expand(B, -1, -1)
            return chroma, h
        return chroma

    # ------------------------------------------------------------------

    def build_rule_hidden_analytic(
        self,
        pc_0:   torch.Tensor,   # (B,) starting pitch class (= key)
        T:      int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build rule_hidden analytically without any sequence input.

        Returns (B, T, d_model=16):
            dims  0-11 : binary chromagram of major triad at bar-phase
            dims 12-15 : one-hot of bar % 4
        """
        B        = pc_0.shape[0]
        t_idx    = torch.arange(T, device=device)
        phase    = self._bar_phase(t_idx)                             # (T,)
        o_cur    = self._offsets.to(device)[phase]                    # (T,)
        cur_root = (pc_0[:, None].to(device) + o_cur[None, :]) % N_ROOTS  # (B, T)
        chroma   = self._root_to_chromagram(cur_root)                 # (B, T, 12)

        h = torch.zeros(B, T, self.d_model, device=device)
        h[:, :, :N_ROOTS] = chroma
        pos_oh = F.one_hot(phase, num_classes=N_POS).float()
        h[:, :, N_ROOTS:] = pos_oh.unsqueeze(0).expand(B, -1, -1)
        return h

    # ------------------------------------------------------------------

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    model = BassTracrRuleModel()
    print(f'TRACR_D_MODEL = {model.TRACR_D_MODEL}  d_model = {model.d_model}')

    # key=0 (C): sequence = [0,5,7,0,0,5,7,0,...]
    idx = torch.tensor([[0, 5, 7, 0, 0, 5, 7, 0]], dtype=torch.long)
    logits, hidden = model(idx, return_hidden=True)
    preds = logits.argmax(-1).squeeze().tolist()
    print(f'input  : {idx.squeeze().tolist()}')
    print(f'preds  : {preds}')
    print(f'expect : [0, 5, 7, 0, 0, 5, 7, 0]  (current pitch class)')
    print(f'hidden shape: {hidden.shape}')
    print(f'All correct: {preds == [0, 5, 7, 0, 0, 5, 7, 0]}')

    # Test CPChordRuleModel
    chord_model = CPChordRuleModel()
    print(f'\nCPChordRuleModel d_model={chord_model.d_model}')
    idx2 = torch.tensor([[0, 5, 7, 0]], dtype=torch.long)  # key=C=0
    chroma, h2 = chord_model(idx2, return_hidden=True)
    print(f'key=C  chromagrams: {chroma.squeeze().tolist()}')
    # t=0: root=0 → C major {0,4,7}; t=1: root=5 → F major {5,9,0}
    # t=2: root=7 → G major {7,11,2}; t=3: root=0 → C major {0,4,7}
    print(f'hidden shape: {h2.shape}')
