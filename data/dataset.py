"""
Sequence dataset for the modulo rule:
  position i: (x + OFFSETS[i % 4]) % 12
where OFFSETS = [0, 5, 7, 0] and x is the starting integer.

Splits mirror the bass CP experiment exactly (vocab=12, same counts):

  Pretrain set  = {0, 5, 9}                           – AR pretraining  (3 starters)
  Finetune set  = {0, 1, 2, 3, 4, 7, 10, 11}          – adapter training (8 starters)

Four-way evaluation partition:
  Pretrain-only  = {5, 9}                    – seen only during pretraining  (2 starters)
  Finetune-only  = {1, 2, 3, 4, 7, 10, 11}  – seen only during finetuning   (7 starters)
  Both           = {0}                       – seen in both stages            (1 starter)
  Neither        = {6, 8}                    – never seen as starters;
                                               tokens {6,11,13,8,13,15} are
                                               covered by training sequences   (2 starters)

Identical to bass CP key splits: pretrain-only={5,9}, finetune-only={1,2,3,4,7,10,11},
both={0}, unseen={6,8}.  Enables direct comparison between the two experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 12
OFFSETS = [0, 5, 7, 0]       # rule offsets per cycle step (period 4)

# Training splits  (matched to bass CP experiment)
AR_TRAIN_STARTERS      = [0, 5, 9]                    # pretrain  (3 starters)
FINETUNE_STARTERS      = [0, 1, 2, 3, 4, 7, 10, 11]   # all seen  (8 starters)
TEST_STARTERS          = [6, 8]                        # never-seen starters

# Backward-compat alias
ADAPTER_TRAIN_STARTERS = FINETUNE_STARTERS
TRAIN_STARTERS         = AR_TRAIN_STARTERS

# Four evaluation categories
EVAL_PRETRAIN_ONLY = [5, 9]                      # seen only in pretrain
EVAL_FINETUNE_ONLY = [1, 2, 3, 4, 7, 10, 11]    # seen only in finetune
EVAL_BOTH          = [0]                         # seen in both stages
EVAL_NEITHER       = [6, 8]                      # never seen


def make_sequence(x: int, n_cycles: int) -> list[int]:
    """Return [t_0, t_1, ..., t_{4n-1}] for n full cycles."""
    return [(x + OFFSETS[i % 4]) % VOCAB_SIZE for i in range(4 * n_cycles)]


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
