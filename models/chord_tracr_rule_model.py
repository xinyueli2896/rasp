"""
Analytical rule model for v6: one chord per bar, ROOT and QUALITY.

v4 and v5 asked the rule model for a root per bar:

    root[b] = (key + OFFSETS[b % 4]) % 12          OFFSETS = [0, 5, 7, 0]

Those offsets are root INTERVALS, and they are identical for major I-IV-V-I
and minor i-iv-v-i -- only the thirds differ. So the root-only rule is blind to
the mode, and the datasets built on it pool both. v6 adds the missing channel:

    qual[b] = QUAL[mode][b % 4]

where `mode` is a second latent, read from the prompt the way the key already
is. Which patterns QUAL should hold is an empirical question that
midi_adapter/census_v6_quality.py answers against the corpus, so they are
constructor arguments, not constants baked into the model. The three the
census counts:

    major            M M M M     constant: the quality channel says nothing
    natural minor    m m m m     one bit per song, "copy the mode"
    harmonic minor   m m M m     the third FLIPS on the V chord

Only the last makes quality vary inside the cycle, which is what forces a model
to consult the rule instead of tracking roots.

Zero parameters, like BassTracrRuleModel, which this leaves untouched.

Residual stream
---------------
rule_mode='period4' (the one the adapter uses), with n modes:

    dims  0-11  one-hot current root            (key + OFFSETS[t % 4]) % 12
    dims 12-13  one-hot current quality         QUAL[mode][t % 4]
    dims 14-25  one-hot next root               t+1
    dims 26-27  one-hot next quality            t+1
    dims 28-31  one-hot phase                   t % 4
    dims 32..   one-hot mode, broadcast         constant along t
                                                d_model = 32 + n

rule_mode='current' drops the two next-* blocks:  d_model = 18 + n.

With two modes that is d_model = 34, against 28 for the bass model's period4.
The adapter's k_proj/v_proj take rule_dim as their input width, so a v6 adapter
cannot load a v5 checkpoint -- v6 is a new experiment, not a continuation. The
frozen-host comparison is unaffected.

On being "Tracr-equivalent"
--------------------------
The construction here is a fixed lookup over (key, mode, phase) -- 12 x n x 4
cells -- which is the shape a Tracr `SequenceMap` over two broadcast values and
a positional value compiles to: one MLP hidden unit per cell, each writing its
cell's one-hot root and quality into the residual stream. Nothing is learned
and nothing is approximated, so the lookup is written directly rather than
simulated with an attention stack. Getting `key` and `mode` broadcast from the
prompt to every position IS an attention operation, and it is the same
seed_broadcast pattern BassTracrRuleModel already implements; here both arrive
as arguments, extracted from the prompt by the caller (see
midi_adapter/census_v6_quality.bar_quality for how the mode is read).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasp_program.sequence_rule import OFFSETS

N_ROOTS = 12
N_POS = len(OFFSETS)                      # 4 bars to a cycle
QUALITIES = ('M', 'm')                    # major third, minor third
N_QUAL = len(QUALITIES)                   # the symbols census_v6_quality uses
_QUAL_IDX = {q: i for i, q in enumerate(QUALITIES)}

# The candidate rules the census counts. A mode name maps to the quality of
# each bar of the cycle.
QUAL_PATTERNS: dict[str, tuple[str, ...]] = {
    'major':          ('M', 'M', 'M', 'M'),
    'natural minor':  ('m', 'm', 'm', 'm'),
    'harmonic minor': ('m', 'm', 'M', 'm'),
}

# Which modes a model carries by default: the pair whose quality varies inside
# the cycle for one of them, so the channel is informative. Replace it with
# whatever the census attests.
DEFAULT_MODES = ('major', 'harmonic minor')


class ChordTracrRuleModel(nn.Module):
    """root[b] = (key + OFFSETS[b%4]) % 12,  qual[b] = QUAL[mode][b%4]."""

    def __init__(
        self,
        modes:     tuple[str, ...] = DEFAULT_MODES,
        rule_mode: str = 'period4',
        patterns:  dict[str, tuple[str, ...]] | None = None,
    ):
        super().__init__()
        if rule_mode not in ('current', 'period4'):
            raise ValueError(f"rule_mode must be 'current' or 'period4', "
                             f'got {rule_mode!r}')
        patterns = patterns or QUAL_PATTERNS
        missing = [m for m in modes if m not in patterns]
        if missing:
            raise ValueError(f'no quality pattern for {missing}; '
                             f'known: {sorted(patterns)}')
        for m in modes:
            p = patterns[m]
            if len(p) != N_POS or any(q not in _QUAL_IDX for q in p):
                raise ValueError(f'pattern for {m!r} must be {N_POS} of '
                                 f'{QUALITIES}, got {p!r}')
        if not modes:
            raise ValueError('at least one mode is required')

        self.rule_mode = rule_mode
        self.modes = tuple(modes)
        self.n_modes = len(self.modes)
        self.patterns = {m: patterns[m] for m in self.modes}

        # --- residual stream layout ---
        off = 0
        self.sl_root = slice(off, off + N_ROOTS);              off += N_ROOTS
        self.sl_qual = slice(off, off + N_QUAL);               off += N_QUAL
        if rule_mode == 'period4':
            self.sl_next_root = slice(off, off + N_ROOTS);     off += N_ROOTS
            self.sl_next_qual = slice(off, off + N_QUAL);      off += N_QUAL
        else:
            self.sl_next_root = self.sl_next_qual = None
        self.sl_phase = slice(off, off + N_POS);               off += N_POS
        self.sl_mode = slice(off, off + self.n_modes);         off += self.n_modes
        self.d_model = off

        self.register_buffer('_offsets', torch.tensor(OFFSETS, dtype=torch.long))
        # (n_modes, N_POS) quality index per mode and phase -- the whole rule.
        self.register_buffer('_qual', torch.tensor(
            [[_QUAL_IDX[q] for q in self.patterns[m]] for m in self.modes],
            dtype=torch.long))

    # ------------------------------------------------------------------
    # The rule itself
    # ------------------------------------------------------------------

    def expected(self, key: torch.Tensor, mode: torch.Tensor, T: int):
        """The rule's answer at every position: (roots, quals), both (B, T).

        key  : (B,) pitch class of the song's tonic
        mode : (B,) index into self.modes
        Also what the data prep and the metrics should score against, so the
        rule lives in exactly one place.
        """
        device = key.device
        t = torch.arange(T, device=device)
        phase = t % N_POS                                        # (T,)
        roots = (key[:, None] + self._offsets.to(device)[phase][None, :]) % N_ROOTS
        quals = self._qual.to(device)[mode][:, phase]             # (B, T)
        return roots, quals

    # ------------------------------------------------------------------
    # Analytic rule_hidden -- the path the adapter uses
    # ------------------------------------------------------------------

    def build_rule_hidden_analytic(
        self,
        pc_0:   torch.Tensor,     # (B,) tonic pitch class, read from the prompt
        mode_0: torch.Tensor,     # (B,) mode index, read from the prompt
        T:      int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """(B, T, d_model), mirroring BassTracrRuleModel's method plus a mode."""
        device = device or pc_0.device
        pc_0 = pc_0.to(device).long()
        mode_0 = mode_0.to(device).long()
        B = pc_0.shape[0]
        t = torch.arange(T, device=device)

        roots, quals = self.expected(pc_0, mode_0, T)
        h = torch.zeros(B, T, self.d_model, device=device)
        h[:, :, self.sl_root] = F.one_hot(roots, N_ROOTS).float()
        h[:, :, self.sl_qual] = F.one_hot(quals, N_QUAL).float()
        if self.rule_mode == 'period4':
            # The next bar's chord, which is what an adapter at position t
            # needs when t's own chord is already sounding.
            nxt_phase = (t + 1) % N_POS
            n_roots = (pc_0[:, None]
                       + self._offsets.to(device)[nxt_phase][None, :]) % N_ROOTS
            n_quals = self._qual.to(device)[mode_0][:, nxt_phase]
            h[:, :, self.sl_next_root] = F.one_hot(n_roots, N_ROOTS).float()
            h[:, :, self.sl_next_qual] = F.one_hot(n_quals, N_QUAL).float()
        h[:, :, self.sl_phase] = F.one_hot(t % N_POS, N_POS).float()[None].expand(B, -1, -1)
        h[:, :, self.sl_mode] = F.one_hot(mode_0, self.n_modes).float()[:, None, :].expand(-1, T, -1)
        return h

    # ------------------------------------------------------------------

    def forward(self, pc: torch.Tensor, mode: torch.Tensor,
                return_hidden: bool = False):
        """pc: (B, T) pitch classes, pc[:, 0] is the tonic. mode: (B,).

        Returns (root_logits (B,T,12), qual_logits (B,T,2)), read out of the
        residual stream's next-* blocks under 'period4' and its current-*
        blocks under 'current' -- the same convention as BassTracrRuleModel,
        whose period4 logits are the next pitch class.
        """
        T = pc.shape[1]
        h = self.build_rule_hidden_analytic(pc[:, 0], mode, T, pc.device)
        if self.rule_mode == 'period4':
            root_logits = h[:, :, self.sl_next_root]
            qual_logits = h[:, :, self.sl_next_qual]
        else:
            root_logits = h[:, :, self.sl_root]
            qual_logits = h[:, :, self.sl_qual]
        if return_hidden:
            return root_logits, qual_logits, h
        return root_logits, qual_logits

    # ------------------------------------------------------------------

    def describe(self) -> str:
        rows = [f'  {m:16s} ' + ' '.join(self.patterns[m]) for m in self.modes]
        return (f'ChordTracrRuleModel(rule_mode={self.rule_mode!r}, '
                f'd_model={self.d_model}, modes={self.n_modes})\n'
                f'  OFFSETS          ' + ' '.join(map(str, OFFSETS)) + '\n'
                + '\n'.join(rows))

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return iter([])


if __name__ == '__main__':
    m = ChordTracrRuleModel()
    print(m.describe())
    key = torch.tensor([0, 4])                 # C and E
    mode = torch.tensor([0, 1])                # major, harmonic minor
    roots, quals = m.expected(key, mode, 8)
    for i in range(2):
        chords = ' '.join(f'{int(r)}{QUALITIES[int(q)]}'
                          for r, q in zip(roots[i], quals[i]))
        print(f'  key {int(key[i]):2d} {m.modes[int(mode[i])]:16s} {chords}')
    h = m.build_rule_hidden_analytic(key, mode, 8)
    print(f'  rule_hidden {tuple(h.shape)}, parameters '
          f'{sum(1 for _ in m.parameters())}')
