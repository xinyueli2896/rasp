"""
Sequence dataset for the modulo rule:
  position i: (x + OFFSETS[i % 3]) % 12
where OFFSETS = [0, 5, 7] and x is the starting integer.

Train/test split is over *starting integers*:
  train_starters  = {0..SPLIT_AT-1}
  test_starters   = {SPLIT_AT..11}
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 12
OFFSETS = [0, 5, 7]          # rule offsets per cycle step

# Three-way split:
#   AR_TRAIN_STARTERS   – used for AR pretraining (AR will be perfect here)
#   ADAPTER_TRAIN_STARTERS – used for adapter training (includes starters
#                            the AR model has NOT seen → gives the adapter
#                            a training signal to trust the rule model)
#   TEST_STARTERS        – held out for evaluation (truly unseen)
AR_TRAIN_STARTERS      = list(range(6))      # 0-5  → AR pretrained
ADAPTER_TRAIN_STARTERS = list(range(8))      # 0-7  → adapter training
                                             #   starters 6-7: AR wrong, rule right
                                             #   starters 0-5: AR right, rule right
TEST_STARTERS          = list(range(8, 12))  # 8-11 → test generalisation

# Backward-compat alias used by evaluation
TRAIN_STARTERS = AR_TRAIN_STARTERS


def make_sequence(x: int, n_cycles: int) -> list[int]:
    """Return [t_0, t_1, ..., t_{3n-1}] for n full cycles."""
    return [(x + OFFSETS[i % 3]) % VOCAB_SIZE for i in range(3 * n_cycles)]


def build_examples(starters: list[int], n_cycles: int, n_seqs_per_starter: int = 1):
    """
    Build (input, target) pairs for teacher-forced next-token prediction.

    Each example is one full sequence.  We slide a window so:
      input  = seq[:-1]
      target = seq[1:]
    """
    inputs, targets = [], []
    for x in starters:
        for _ in range(n_seqs_per_starter):
            seq = make_sequence(x, n_cycles)
            inputs.append(seq[:-1])
            targets.append(seq[1:])
    return inputs, targets


class SequenceDataset(Dataset):
    def __init__(self, starters: list[int], n_cycles: int = 8, n_seqs_per_starter: int = 100):
        inputs, targets = build_examples(starters, n_cycles, n_seqs_per_starter)
        self.inputs  = torch.tensor(inputs,  dtype=torch.long)   # (N, seq_len-1)
        self.targets = torch.tensor(targets, dtype=torch.long)   # (N, seq_len-1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(
    batch_size: int = 64,
    n_cycles: int = 8,
    n_seqs_per_starter: int = 100,
    train_starters: list[int] | None = None,
):
    if train_starters is None:
        train_starters = AR_TRAIN_STARTERS
    train_ds = SequenceDataset(train_starters, n_cycles, n_seqs_per_starter)
    test_ds  = SequenceDataset(TEST_STARTERS,  n_cycles, n_seqs_per_starter)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_adapter_dataloaders(
    batch_size: int = 64,
    n_cycles: int = 8,
    n_seqs_per_starter: int = 100,
):
    """Returns loaders with ADAPTER_TRAIN_STARTERS for train and TEST_STARTERS for test."""
    return get_dataloaders(batch_size, n_cycles, n_seqs_per_starter,
                           train_starters=ADAPTER_TRAIN_STARTERS)


if __name__ == "__main__":
    # Quick sanity check
    for x in [0, 7, 11]:
        seq = make_sequence(x, 3)
        print(f"x={x}: {seq}")
    train_loader, test_loader = get_dataloaders()
    inp, tgt = next(iter(train_loader))
    print(f"\nBatch shapes: input={inp.shape}, target={tgt.shape}")
    print(f"Sample input:  {inp[0].tolist()}")
    print(f"Sample target: {tgt[0].tolist()}")
