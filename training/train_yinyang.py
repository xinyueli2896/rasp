"""
Train the Yin-Yang model.

The AR transformer (Yang) is either LoRA-adapted or fully frozen.
The rule model (Yin) is always frozen.
Only the yinyang_attn cross-attention modules (and LoRA weights if enabled) train.

Usage:
    python training/train_yinyang.py --force_fallback
    python training/train_yinyang.py --force_fallback --no_lora
"""

from __future__ import annotations

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset  import (get_adapter_dataloaders,
                            AR_TRAIN_STARTERS, ADAPTER_TRAIN_STARTERS,
                            TEST_STARTERS, VOCAB_SIZE)
from models.yinyang_model import build_yinyang_model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'AR pretrain starters  : {AR_TRAIN_STARTERS}')
    print(f'Adapter train starters: {ADAPTER_TRAIN_STARTERS}')
    print(f'Test starters         : {TEST_STARTERS}')

    train_loader, test_loader = get_adapter_dataloaders(
        batch_size         = args.batch_size,
        n_cycles           = args.n_cycles,
        n_seqs_per_starter = args.n_seqs_per_starter,
    )

    model = build_yinyang_model(
        ar_ckpt_path   = args.ar_ckpt,
        max_seq_len    = args.n_cycles * 4 + 10,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        n_heads        = args.n_heads,
        rule_d_model   = args.d_model,
        adapter_rank   = args.adapter_rank,
        n_skip         = args.n_skip,
        use_lora       = args.use_lora,
        force_fallback = args.force_fallback,
        device         = str(device),
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Trainable: {n_trainable:,}   Frozen: {n_frozen:,}')
    print(f'yinyang_attn modules: {len(model.yinyang_attn)}  (one per {args.n_skip} AR layers)')

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        # rule model has no params so .eval() is a no-op, but keeps intent clear
        model.rule_model.eval()

        total_loss, total_correct, total_tokens = 0.0, 0, 0

        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)

            logits = model(inp)
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss    += loss.item() * tgt.numel()
            total_correct += (logits.argmax(-1) == tgt).sum().item()
            total_tokens  += tgt.numel()

        scheduler.step()
        train_loss = total_loss / total_tokens
        train_acc  = total_correct / total_tokens

        # --- Eval ---
        model.eval()
        with torch.no_grad():
            test_correct, test_tokens = 0, 0
            for inp, tgt in test_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)
                test_correct += (logits.argmax(-1) == tgt).sum().item()
                test_tokens  += tgt.numel()
        test_acc = test_correct / test_tokens

        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f'Epoch {epoch:4d}/{args.epochs}  '
                f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
                f'test_acc(unseen)={test_acc:.4f}'
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            path = os.path.join(args.ckpt_dir, 'yinyang.pt')
            # Save only trainable params (yinyang_attn + LoRA weights)
            trainable_state = {k: v for k, v in model.state_dict().items()
                               if any(k.startswith(prefix) for prefix in
                                      ('yinyang_attn',))}
            if args.use_lora:
                # also save LoRA weights (stored inside ar_model)
                for k, v in model.ar_model.state_dict().items():
                    if 'lora_' in k:
                        trainable_state[f'ar_model.{k}'] = v
            torch.save(trainable_state, path)

    print(f'\nBest test accuracy on unseen starters: {best_test_acc:.4f}')
    print(f'Saved checkpoint → {os.path.join(args.ckpt_dir, "yinyang.pt")}')


def get_args():
    p = argparse.ArgumentParser(description='Train Yin-Yang model')
    p.add_argument('--ar_ckpt',            type=str,   default='checkpoints/ar_transformer.pt')
    p.add_argument('--epochs',             type=int,   default=100)
    p.add_argument('--batch_size',         type=int,   default=64)
    p.add_argument('--lr',                 type=float, default=1e-4)
    p.add_argument('--d_model',            type=int,   default=128)
    p.add_argument('--n_layers',           type=int,   default=4)
    p.add_argument('--n_heads',            type=int,   default=4)
    p.add_argument('--adapter_rank',       type=int,   default=32)
    p.add_argument('--n_skip',             type=int,   default=2,
                   help='Inject rule hidden states every n_skip AR layers')
    p.add_argument('--n_cycles',           type=int,   default=8)
    p.add_argument('--n_seqs_per_starter', type=int,   default=200)
    p.add_argument('--log_every',          type=int,   default=10)
    p.add_argument('--ckpt_dir',           type=str,   default='checkpoints')
    p.add_argument('--use_lora',           action='store_true', default=True)
    p.add_argument('--no_lora',            action='store_false', dest='use_lora')
    p.add_argument('--force_fallback',     action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    train(get_args())
