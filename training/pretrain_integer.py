#!/usr/bin/env python3
"""Pretrain the AR transformer on pool A, then score all four cells.

    python -m training.pretrain_integer
    python -m training.pretrain_integer --epochs 400 --seed 1

WHAT THIS PRODUCES
------------------
checkpoints/ar_integer_seed<N>.pt, and the baseline row of the results
table -- the row every adapter number is read against.

WHY THE METRIC IS GENERATIVE, NOT TEACHER-FORCED
------------------------------------------------
Teacher-forced accuracy is what the loss optimises, and it flatters the
model: at every position it is handed the correct prefix, so a model that
has merely memorised "after 5 comes 10" scores well without ever having to
carry a starter through 31 steps. The claim under test is that the model
APPLIES a rule, so the metric matches `report.tex`:

    RFA(x): prompt with the single token [x], generate 4*n_cycles - 1 = 31
            tokens autoregressively, report the fraction matching the rule.

Both are printed, because the gap between them is itself informative: high
teacher-forced with low RFA means the model is local-pattern matching and
drifts once it has to stand on its own output.

THE QUESTION THIS RUN ANSWERS
-----------------------------
Not "is the AR good" -- it is whether the split leaves the adapter anything
to do. The rule is translation-equivariant in x, so if 5 of 12 starters is
enough for the AR to learn the RULE rather than the starters, it will
generalise to {6, 8} unaided and every later adapter number is
uninterpretable. The run ends by saying which happened; see
`data.integer_dataset.ar_baseline_is_informative`.

Read the cells as:

    both           control; should be ~1.0 or training did not converge
    pretrain_only  the AR's own starters, held out of adapter training
    finetune_only  AR has never seen these; expected LOW
    unseen         the headline; expected LOW, and must stay low
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.integer_dataset import (  # noqa: E402
    N_CYCLES, PRETRAIN_STARTERS, SEQ_LEN, ar_baseline_is_informative,
    get_dataloaders, validate_split,
)
from models.transformer import AutoregressiveTransformer  # noqa: E402
from rasp_program.next_token_rule import VOCAB_SIZE, sequence  # noqa: E402

CELLS = ('both', 'pretrain_only', 'finetune_only', 'unseen')


@torch.no_grad()
def rule_following_acc(model, starters, n_cycles, device) -> tuple[float, float]:
    """(RFA, teacher-forced acc) for a list of starters.

    RFA generates from the one-token prompt [x]; teacher-forced scores
    next-token prediction given the true prefix. Greedy decoding -- the rule
    is deterministic, so sampling would only add noise to a quantity that
    has a single right answer.
    """
    T = 4 * n_cycles
    truth = torch.as_tensor(sequence(starters, T), device=device)     # (B, T)
    model.eval()

    # generative: start from token 0 only
    ctx = truth[:, :1].clone()
    for _ in range(T - 1):
        ctx = torch.cat([ctx, model(ctx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    rfa = float((ctx[:, 1:] == truth[:, 1:]).float().mean())

    # teacher-forced: correct prefix at every step
    logits = model(truth[:, :-1])
    tf = float((logits.argmax(-1) == truth[:, 1:]).float().mean())
    return rfa, tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_cycles', type=int, default=N_CYCLES)
    p.add_argument('--n_repeats', type=int, default=200,
                   help='Copies of each sequence per epoch. The sequences are '
                        'deterministic, so this only sets the epoch length; it '
                        'adds no information.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt_dir', type=str, default='checkpoints')
    p.add_argument('--ckpt_name', type=str, default=None)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--device', type=str, default=None)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    info = validate_split(n_cycles=a.n_cycles)
    cells = info['cells']
    print(f'device       : {dev}')
    print(f'A  pretrain  : {info["pretrain"]}')
    print(f'B  adapter   : {info["adapter"]}   (not trained on here)')
    for k in CELLS:
        print(f'  {k:<14} {cells[k]}')
    if info['per_pool_gap']:
        print(f'  per-pool coverage gap: {info["per_pool_gap"]}')
    print()

    train_loader, _ = get_dataloaders(
        'pretrain', batch_size=a.batch_size, n_cycles=a.n_cycles,
        n_repeats=a.n_repeats)

    model = AutoregressiveTransformer(
        vocab_size=VOCAB_SIZE, max_seq_len=4 * a.n_cycles + 8,
        d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads).to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    print(f'model        : d_model={a.d_model} layers={a.n_layers} '
          f'heads={a.n_heads}  {n_par:,} params\n')

    crit  = nn.CrossEntropyLoss()
    optim_ = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = CosineAnnealingLR(optim_, T_max=a.epochs)

    for ep in range(1, a.epochs + 1):
        model.train()
        tot, hits, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            optim_.zero_grad(); loss.backward(); optim_.step()
            tot  += loss.item() * y.numel()
            hits += int((logits.argmax(-1) == y).sum()); n += y.numel()
        sched.step()
        if ep % a.log_every == 0 or ep == 1 or ep == a.epochs:
            rfa, _ = rule_following_acc(model, info['pretrain'], a.n_cycles, dev)
            print(f'  epoch {ep:4d}  loss {tot/n:.4f}  train_tf {hits/n:.4f}  '
                  f'train_RFA {rfa:.4f}  lr {sched.get_last_lr()[0]:.2e}')

    # -- the baseline row --------------------------------------------------
    print(f'\n{"cell":<16}{"starters":<24}{"n":>3}{"RFA":>9}{"teacher-forced":>16}')
    print('-' * 68)
    acc = {}
    for k in CELLS:
        rfa, tf = rule_following_acc(model, cells[k], a.n_cycles, dev)
        acc[k] = rfa
        print(f'{k:<16}{str(cells[k]):<24}{len(cells[k]):>3}{rfa:>9.3f}{tf:>16.3f}')
    print(f'\n  RFA = generate 4*{a.n_cycles}-1 = {4*a.n_cycles-1} tokens from the '
          f'one-token prompt [x], greedy.')
    print(f'  chance = {1/VOCAB_SIZE:.3f}')

    print(f'\n{ar_baseline_is_informative(acc)}')

    name = a.ckpt_name or f'ar_integer_seed{a.seed}.pt'
    os.makedirs(a.ckpt_dir, exist_ok=True)
    path = os.path.join(a.ckpt_dir, name)
    torch.save({'state_dict': model.state_dict(),
                'args': vars(a), 'split': info, 'rfa': acc}, path)
    print(f'\nsaved {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
