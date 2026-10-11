#!/usr/bin/env python3
"""Train the cross-attention adapter on pool B, and the FT baseline.

    python -m training.finetune_adapter_integer                  # adapter
    python -m training.finetune_adapter_integer --mode finetune  # FT baseline

Both modes train on pool B and score the same four cells, so the rows line
up against the AR baseline from training/pretrain_integer.py:

    Pre   AR alone                frozen, trained on A
    FT    AR continued on B       all parameters trainable, no rule model
    YY    adapter over frozen AR  only the cross-attention trains

FT is the catastrophic-forgetting control. It should do well on
finetune_only and collapse on pretrain_only; the adapter's claim is that it
gets the former without the latter. Without this row "the adapter learned
B" is not evidence of anything, because plain fine-tuning also learns B.

n_skip=1 injects after every AR block (4 adapters). The chord experiment's
sweep found more injection points monotonically better, and at this scale
the adapter is small either way.
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

from data.integer_dataset import N_CYCLES, get_dataloaders, validate_split
from models.integer_adapter import SLICES, build
from rasp_program.next_token_rule import VOCAB_SIZE, sequence
from training.pretrain_integer import CELLS

from models.transformer import AutoregressiveTransformer  # noqa: E402


@torch.no_grad()
def rfa(model, starters, n_cycles, device):
    """Generative rule-following accuracy, and teacher-forced, per report.tex."""
    T = 4 * n_cycles
    truth = torch.as_tensor(sequence(starters, T), device=device)
    model.eval()
    ctx = truth[:, :1].clone()
    for _ in range(T - 1):
        ctx = torch.cat([ctx, model(ctx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    gen = float((ctx[:, 1:] == truth[:, 1:]).float().mean())
    tf  = float((model(truth[:, :-1]).argmax(-1) == truth[:, 1:]).float().mean())
    return gen, tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['adapter', 'finetune'], default='adapter')
    p.add_argument('--ar_ckpt', type=str, default='checkpoints/ar_integer_seed0.pt')
    p.add_argument('--rule_ckpt', type=str, default=None)
    p.add_argument('--rule_slice', choices=list(SLICES), default='program')
    p.add_argument('--n_skip', type=int, default=1)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--n_cycles', type=int, default=N_CYCLES)
    p.add_argument('--n_repeats', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt_dir', type=str, default='checkpoints')
    p.add_argument('--ckpt_name', type=str, default=None)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--device', type=str, default=None)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    info = validate_split(n_cycles=a.n_cycles)
    cells = info['cells']

    print(f'device   : {dev}   mode: {a.mode}')
    print(f'A pretrain: {info["pretrain"]}   B adapter: {info["adapter"]}')

    if a.mode == 'adapter':
        model = build(a.ar_ckpt, a.rule_ckpt, n_skip=a.n_skip,
                      embed_dim=a.embed_dim, n_heads=a.n_heads,
                      rule_slice=a.rule_slice,
                      max_seq_len=4 * a.n_cycles + 8, device=dev)
        print(f'rule_slice: {a.rule_slice} -> {SLICES[a.rule_slice]}  '
              f'({model.rule_dims.numel()} dims)')
        print(f'adapters : {len(model.adapters)} x embed_dim {a.embed_dim}, '
              f'n_skip {a.n_skip}')
        print(f'trainable: {model.n_trainable():,}   frozen: {model.n_frozen():,}'
              f'   ({100 * model.n_trainable() / model.n_frozen():.1f}% of frozen)')
    else:
        blob = torch.load(a.ar_ckpt, map_location='cpu', weights_only=False)
        ab = blob['args']
        model = AutoregressiveTransformer(
            vocab_size=VOCAB_SIZE, max_seq_len=4 * a.n_cycles + 8,
            d_model=ab['d_model'], n_layers=ab['n_layers'],
            n_heads=ab['n_heads']).to(dev)
        model.load_state_dict(blob['state_dict'])
        print(f'trainable: {sum(q.numel() for q in model.parameters()):,} '
              f'(ALL of the AR -- this is the forgetting control)')

    # The AR baseline this is measured against, if the checkpoint carries it.
    base = torch.load(a.ar_ckpt, map_location='cpu',
                      weights_only=False).get('rfa')
    if base:
        print('AR baseline: ' + '  '.join(f'{k}={base[k]:.3f}' for k in CELLS))
    print()

    train_loader, _ = get_dataloaders('adapter', batch_size=a.batch_size,
                                      n_cycles=a.n_cycles,
                                      n_repeats=a.n_repeats)
    params = [q for q in model.parameters() if q.requires_grad]
    crit, opt = nn.CrossEntropyLoss(), optim.AdamW(
        params, lr=a.lr, weight_decay=a.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=a.epochs)

    for ep in range(1, a.epochs + 1):
        model.train()
        tot, hits, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * y.numel()
            hits += int((logits.argmax(-1) == y).sum()); n += y.numel()
        sched.step()
        if ep % a.log_every == 0 or ep in (1, a.epochs):
            g, _ = rfa(model, info['adapter'], a.n_cycles, dev)
            print(f'  epoch {ep:4d}  loss {tot/n:.4f}  train_tf {hits/n:.4f}  '
                  f'train_RFA {g:.4f}  lr {sched.get_last_lr()[0]:.2e}')

    print(f'\n{"cell":<16}{"starters":<24}{"n":>3}{"RFA":>9}{"vs AR":>9}'
          f'{"teacher-forced":>16}')
    print('-' * 77)
    out = {}
    for k in CELLS:
        g, tf = rfa(model, cells[k], a.n_cycles, dev)
        out[k] = g
        delta = f'{g - base[k]:+.3f}' if base else '--'
        print(f'{k:<16}{str(cells[k]):<24}{len(cells[k]):>3}{g:>9.3f}'
              f'{delta:>9}{tf:>16.3f}')
    print(f'\n  chance = {1/VOCAB_SIZE:.3f}')
    print('  pretrain_only is the forgetting cell: the FT baseline is expected')
    print('  to collapse there while the adapter holds it, since the adapter')
    print('  cannot alter the frozen AR weights at all.')
    print('  unseen is the headline. Neither pool contains {6, 8}.')

    name = a.ckpt_name or f'integer_{a.mode}_{a.rule_slice}_seed{a.seed}.pt'
    os.makedirs(a.ckpt_dir, exist_ok=True)
    path = os.path.join(a.ckpt_dir, name)
    torch.save({'state_dict': model.state_dict(), 'args': vars(a),
                'split': info, 'rfa': out, 'ar_baseline': base}, path)
    print(f'\nsaved {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
