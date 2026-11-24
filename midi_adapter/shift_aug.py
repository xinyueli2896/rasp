"""
Pitch-shift augmentation in the training loop, the way v0.42 was pretrained.

Upstream (music-x-lab/midi-function-alignment) does this in two pieces:

  preprocess_large_midi_dataset   per song, pitch_shift_range =
                                  [-min_pitch, 127 - max_pitch] over its
                                  non-drum notes
  FramedDataset.__iter__          clamps that to [-5, 6] and draws one integer
                                  shift per window, uniformly; val/test get 0
  preprocess                      adds the shift to every non-drum pitch token

Nothing is transposed ahead of time: the stored data stays at its own key and
each window is moved as it is drawn. This module does the same, computed when
the dataset loads -- no data rebuild -- with one restriction the held-out
split needs: a shift is skipped if it would move the song's tonic to a key
(v4/v5) or tonic chord (v6) that is not in this file's split. A C window
shifted +6 IS an F# window, so without that, finetuning would train on the
held-out keys.

The split's labels are read from the file itself: <data>.chords.pt (v6, 24
tonic chords; a shift keeps the mode) or <data>.keys.pt (v4/v5, 12 roots).
Every label that occurs in the file is allowed, so a finetune file never
reaches a key that only the unseen file holds.

Sampling is uniform over the allowed shifts inside the clamped range, i.e.
upstream's draw conditioned on staying in the split. Shift 0 is always
allowed.
"""

from __future__ import annotations

import os

import torch

SHIFT_LO, SHIFT_HI = -5, 6                     # FramedDataset's clamp
SHIFTS = torch.arange(SHIFT_LO, SHIFT_HI + 1)  # column j  <->  shift j - 5
ZERO = -SHIFT_LO


def song_pitch_ranges(data: torch.Tensor, start: torch.Tensor,
                      length: torch.Tensor) -> torch.Tensor:
    """(N, 2) upstream pitch_shift_range per song: [-min_pitch, 127 - max_pitch].

    data is the concatenated (T, 4 * max_polyphony) tensor; a note is live
    where its pitch is not the 255 pad and its program is neither 254 (EOS)
    nor 127 (drums), exactly the notes upstream's min/max runs over.
    """
    d = data.reshape(data.shape[0], -1, 4)
    prog, pitch = d[..., 0].long(), d[..., 1].long()
    live = (pitch != 255) & (prog != 254) & (prog != 127)
    lo_p = torch.where(live, pitch, torch.full_like(pitch, 127)).amin(1)  # (T,)
    hi_p = torch.where(live, pitch, torch.zeros_like(pitch)).amax(1)
    out = torch.zeros(len(length), 2, dtype=torch.long)
    for i, (s, n) in enumerate(zip(start.tolist(), length.tolist())):
        mn, mx = int(lo_p[s:s + n].min()), int(hi_p[s:s + n].max())
        if mn > mx:                       # no live note at all: do not move it
            continue
        out[i, 0], out[i, 1] = -mn, 127 - mx
    return out


def load_labels(file_path: str):
    """(labels per song, 'chord' | 'key'), from the files the builders write."""
    stem = file_path[:-3]
    if os.path.exists(stem + '.chords.pt'):
        return torch.load(stem + '.chords.pt', weights_only=True).long(), 'chord'
    if os.path.exists(stem + '.keys.pt'):
        return torch.load(stem + '.keys.pt', weights_only=True).long(), 'key'
    raise SystemExit(f'--shift_aug needs {stem}.chords.pt or {stem}.keys.pt to '
                     f'know which keys this split holds')


def shifted(labels: torch.Tensor, kind: str) -> torch.Tensor:
    """(N, 12) the label each song lands on under each shift -5..6."""
    if kind == 'chord':                   # 0..11 major, 12..23 minor
        root, minor = labels % 12, labels // 12
        return (root[:, None] + SHIFTS[None, :]) % 12 + 12 * minor[:, None]
    return (labels[:, None] + SHIFTS[None, :]) % 12


def allowed_mask(data, start, length, file_path) -> tuple[torch.Tensor, str]:
    """(N, 12) bool: shift j - 5 is inside the song's clamped upstream range
    and lands on a label this file holds. Plus a one-line summary."""
    rng = song_pitch_ranges(data, start, length).clamp(SHIFT_LO, SHIFT_HI)
    in_range = (SHIFTS[None, :] >= rng[:, :1]) & (SHIFTS[None, :] <= rng[:, 1:])
    labels, kind = load_labels(file_path)
    if len(labels) != len(length):
        raise SystemExit(f'{file_path}: {len(labels)} labels for {len(length)} songs')
    held = set(labels.tolist())
    land = shifted(labels, kind)
    in_split = torch.tensor([[int(x) in held for x in row] for row in land.tolist()])
    mask = in_range & in_split
    mask[:, ZERO] = True
    per = mask.sum(1).float()
    note = (f'{kind} labels {len(held)}; allowed shifts per song mean '
            f'{per.mean():.1f} of 12, min {int(per.min())}, max {int(per.max())}')
    return mask, note


def sample(mask: torch.Tensor) -> torch.Tensor:
    """One shift per row, uniform over that row's allowed shifts."""
    return torch.multinomial(mask.float(), 1).squeeze(1) - ZERO
