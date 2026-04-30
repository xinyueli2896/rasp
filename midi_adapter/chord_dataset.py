"""
ChordConditionedDataset — mirrors FramedDataset but also yields
bar-level chord token windows alongside each MIDI window.

Paired files required
---------------------
  data/rwc_cp16_v2.pt               (MIDI data, as produced by preprocess_midi)
  data/rwc_cp16_v2.length.pt        (song lengths in subbeats)
  data/rwc_cp16_v2.pitch_shift_range.pt
  data/rwc_cp16_v2.bar_chords.pt    (produced by preprocess_chords.py)

Each yielded batch is a tuple:
  (midi_data, pitch_shift, chord_tokens)

  midi_data    : (batch, target_length, subseq_len)  int16 / uint8
  pitch_shift  : (batch,)                            int64
  chord_tokens : (batch, n_bars_per_window)          int64
                 where n_bars_per_window = target_length // subbeats_per_bar + 1
                 (one extra bar to cover non-bar-aligned windows; the adapter's
                 causal mask prevents out-of-window bars from being attended to)
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from midi_adapter.chord_tokenizer import NO_CHORD_TOKEN

SUBBEATS_PER_BAR = 16


class ChordConditionedDataset(IterableDataset):
    """
    Drop-in replacement for FramedDataset that additionally returns
    bar-level chord tokens aligned with each sampled MIDI window.

    Parameters
    ----------
    midi_path        : path to the preprocessed MIDI .pt file
    chord_path       : path to the bar_chords .pt file (list of 1-D int16 tensors)
    target_length    : window length in subbeats  (e.g. 384)
    batch_size       : number of windows per batch
    split            : 'all' | 'train' | 'val' | 'test'
    split_ratio      : every split_ratio-th song is val (idx%split==1) or test (idx%split==0)
    sample_step      : stride for sampling window start positions
    random_order     : shuffle songs each epoch
    repeat           : loop indefinitely (False = one pass)
    subbeats_per_bar : 16 for 4/4 at beat_div=4
    """

    def __init__(
        self,
        midi_path:        str,
        chord_path:       str,
        target_length:    int,
        batch_size:       int,
        split:            str  = 'all',
        split_ratio:      int  = 10,
        sample_step:      int  = 1,
        random_order:     bool = True,
        repeat:           bool = True,
        subbeats_per_bar: int  = SUBBEATS_PER_BAR,
    ):
        self.midi_path        = midi_path
        self.chord_path       = chord_path
        self.target_length    = target_length
        self.batch_size       = batch_size
        self.sample_step      = sample_step
        self.random_order     = random_order
        self.repeat           = repeat
        self.subbeats_per_bar = subbeats_per_bar

        # Number of bars in each window.  +1 to cover non-bar-aligned starts
        # (a window starting at subbeat k, k % 16 != 0, spans one extra bar).
        self.n_bars_per_window = target_length // subbeats_per_bar + 1

        # --- load length metadata (lightweight, always in memory) ---
        length_path = midi_path[:-3] + '.length.pt'
        self.length = torch.load(length_path, weights_only=True)
        self.start  = torch.cumsum(self.length, dim=0) - self.length

        is_valid      = self.length >= target_length
        song_indices  = torch.arange(len(self.start))

        if split == 'all':
            self.valid_indices = song_indices[is_valid]
        elif split == 'train':
            self.valid_indices = song_indices[is_valid & (song_indices % split_ratio > 1)]
        elif split == 'val':
            self.valid_indices = song_indices[is_valid & (song_indices % split_ratio == 1)]
        elif split == 'test':
            self.valid_indices = song_indices[is_valid & (song_indices % split_ratio == 0)]
        else:
            raise ValueError(f'Unknown split: {split!r}')

        self.split = split
        print(
            f'ChordConditionedDataset {midi_path} split={split}: '
            f'{len(self.valid_indices)} valid songs'
        )

        # data loaded lazily in __iter__
        self.data        = None
        self.bar_chords  = None
        self.pitch_shift_range = None

    # ------------------------------------------------------------------
    def __iter__(self):
        if self.data is None:
            self.data       = torch.load(self.midi_path, weights_only=True)
            self.bar_chords = torch.load(self.chord_path, weights_only=True)
            self.pitch_shift_range = torch.load(
                self.midi_path[:-3] + '.pitch_shift_range.pt', weights_only=True
            ).reshape(-1, 2)
            # clamp pitch shift
            self.pitch_shift_range[self.pitch_shift_range[:, 0] < -5, 0] = -5
            self.pitch_shift_range[self.pitch_shift_range[:, 1] > 6,  1] = 6
            if self.split in ('val', 'test'):
                self.pitch_shift_range = torch.zeros_like(self.pitch_shift_range)
            print(f'Data loaded: {self.midi_path}')

        while True:
            order = (torch.randperm(len(self.valid_indices)) if self.random_order
                     else torch.arange(len(self.valid_indices)))

            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_idx  = order[i:i + self.batch_size]
                raw_ids    = self.valid_indices[batch_idx]          # song indices

                ps_range   = self.pitch_shift_range[raw_ids]
                starts     = (
                    torch.floor(
                        torch.rand(len(raw_ids))
                        * (self.length[raw_ids] - self.target_length)
                        / self.sample_step
                    ).long() * self.sample_step
                    + self.start[raw_ids]
                )

                # ---- MIDI window ----
                idx_matrix = (
                    torch.arange(self.target_length).unsqueeze(0)
                    + starts.unsqueeze(1)
                )                                                    # (B, target_length)
                midi_batch = self.data[idx_matrix]

                # ---- pitch shift ----
                ps_lo, ps_hi = ps_range[:, 0], ps_range[:, 1]
                pitch_shift = (
                    torch.floor(torch.rand(len(raw_ids)) * (ps_hi - ps_lo + 1)).long()
                    + ps_lo
                )

                # ---- chord window ----
                # For each song s and window start `start`, the relative subbeat
                # offset within the song is `start - self.start[s]`.
                # The first bar of the window is `rel_start // subbeats_per_bar`.
                chord_windows = []
                for j, s in enumerate(raw_ids.tolist()):
                    rel_start = (starts[j] - self.start[s]).item()
                    bar_start = rel_start // self.subbeats_per_bar
                    song_chords = self.bar_chords[s].long()          # 1-D tensor

                    end = bar_start + self.n_bars_per_window
                    if end <= len(song_chords):
                        window = song_chords[bar_start:end]
                    else:
                        # pad with NO_CHORD_TOKEN if we fall off the end
                        window = song_chords[bar_start:]
                        pad_len = self.n_bars_per_window - len(window)
                        window = torch.cat([
                            window,
                            torch.full((pad_len,), NO_CHORD_TOKEN, dtype=torch.long),
                        ])
                    chord_windows.append(window)

                chord_batch = torch.stack(chord_windows)             # (B, n_bars_per_window)

                yield midi_batch, pitch_shift, chord_batch

            if not self.repeat:
                break
