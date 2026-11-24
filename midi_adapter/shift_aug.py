"""
Pitch-shift augmentation that never crosses a key split.

v0.42 was pretrained with FramedDataset's augmentation: every song carries a
pitch_shift_range [lo, hi], clamped to [-5, 6], and each training window is
transposed by an integer drawn uniformly from that range, applied to every
non-drum pitch in `preprocess` (val and test windows are never shifted).

v4-v6 switched that off ([0, 0] everywhere) because a shifted window is a
window in a different key -- a C window shifted +6 IS an F# window -- so the
augmentation would leak the held-out keys into finetuning. This module puts
it back with that one hole closed: the per-song range becomes a per-song MASK
over the same twelve shifts, -5..6, and a shift is allowed only if

  1. every non-drum pitch stays inside MIDI range 0..127 after it, which is
     what the upstream range encodes, and
  2. the shifted song's label (its key for v4/v5, its tonic chord for v6)
     is in the song's OWN split -- so finetune songs move only among the
     finetune keys, never into the unseen (or pretrain-only) ones.

Shift 0 is always allowed. Sampling is uniform over the allowed shifts, so
with no split to respect it reduces exactly to v0.42's uniform draw over
[lo, hi]. The data builders write the mask as <prefix>_<split>.shift_mask.pt;
the trainers read it only when passed --shift_aug.
"""

from __future__ import annotations

import torch

SHIFT_LO, SHIFT_HI = -5, 6                     # FramedDataset's clamp
SHIFTS = list(range(SHIFT_LO, SHIFT_HI + 1))   # index i  <->  shift i - 5
N_SHIFTS = len(SHIFTS)
ZERO = -SHIFT_LO


def pitch_range(data: torch.Tensor, max_polyphony: int) -> tuple[int, int] | None:
    """(min, max) non-drum pitch of a song, or None if it has none.

    Same live-note test as prepare_v4_data.transpose: 255 is a pad pitch, 254
    an EOS program, 127 the drum program (whose 'pitch' is a kit piece).
    """
    d = data.reshape(-1, max_polyphony, 4)
    prog, pitch = d[..., 0], d[..., 1]
    live = (pitch != 255) & (prog != 254) & (prog != 127)
    p = pitch[live].to(torch.int16)
    if p.numel() == 0:
        return None
    return int(p.min()), int(p.max())


def shift_mask(songs: list[dict], label_of, shifted_label, allowed: set,
               max_polyphony: int) -> torch.Tensor:
    """(N, 12) bool: which of the shifts -5..6 each song may take.

    label_of(song)              -> the song's label (key or tonic chord)
    shifted_label(label, shift) -> that label after transposing by shift
    allowed                     -> the labels of the split being written
    """
    rows = torch.zeros(len(songs), N_SHIFTS, dtype=torch.bool)
    for i, s in enumerate(songs):
        rng = pitch_range(s['data'], max_polyphony)
        lab = label_of(s)
        for j, k in enumerate(SHIFTS):
            in_range = rng is None or (rng[0] + k >= 0 and rng[1] + k <= 127)
            rows[i, j] = in_range and shifted_label(lab, k) in allowed
        rows[i, ZERO] = True
    return rows


def summarize(mask: torch.Tensor) -> str:
    """One line for the build log: how much augmentation each song gets."""
    if mask.numel() == 0:
        return 'no songs'
    per = mask.sum(1).float()
    return (f'allowed shifts per song: mean {per.mean():.1f} of {N_SHIFTS}, '
            f'min {int(per.min())}, max {int(per.max())}')


def sample(mask: torch.Tensor) -> torch.Tensor:
    """One shift per row, uniform over that row's allowed shifts."""
    return torch.multinomial(mask.float(), 1).squeeze(1) - ZERO
