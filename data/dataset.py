"""
Sequence dataset for the modulo rule:
  position i: (x + OFFSETS[i % 4]) % 24
where OFFSETS = [0, 5, 7, 0] and x is the starting integer.

Starting-integer splits (vocab = 0-23):

  Pretrain set A = {0,1,2,3,4,5}      – AR pretraining
  Finetune set B = {2,3,4,5,6,7,8,9,10,11} – fine-tuning (with or without rule alignment)

Four-way evaluation partition:
  Pretrain-only  (A \ B) = {0, 1}              – seen only during pretraining
  Finetune-only  (B \ A) = {6,7,8,9,10,11}    – seen only during fine-tuning (AR wrong on all)
  Both           (A ∩ B) = {2, 3, 4, 5}       – seen in both stages
  Neither                = {12,13,14,15}       – never seen during any training

Expanding from VOCAB_SIZE=12 to 24 and adding starters {6-11} to finetune gives the
Yinyang adapter 6 starters where AR is wrong (vs 2 before), providing much more
signal to learn a generalizable rule-correction mechanism.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 24
OFFSETS = [0, 5, 7, 0]       # rule offsets per cycle step (period 4)

# Training splits
AR_TRAIN_STARTERS      = list(range(6))          # 0-5   → AR pretrain (set A)
FINETUNE_STARTERS      = list(range(2, 12))       # 2-11  → fine-tune set B
                                                  #   {6-11}: AR wrong, rule right
                                                  #   {2-5}:  AR right, rule right
TEST_STARTERS          = list(range(12, 16))      # 12-15 → never-seen test set

# Backward-compat aliases
ADAPTER_TRAIN_STARTERS = FINETUNE_STARTERS
TRAIN_STARTERS         = AR_TRAIN_STARTERS

# Four evaluation categories
EVAL_PRETRAIN_ONLY = [0, 1]                    # A \ B : seen only in pretrain
EVAL_FINETUNE_ONLY = [6, 7, 8, 9, 10, 11]     # B \ A : seen only in finetune (AR wrong)
EVAL_BOTH          = [2, 3, 4, 5]             # A ∩ B : seen in both stages
EVAL_NEITHER       = [12, 13, 14, 15]         # complement: seen in neither


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
