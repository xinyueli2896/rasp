"""
Fine-tune the pretrained AR transformer on FINETUNE_STARTERS (set B = {2..7})
*without* any rule alignment.

This produces the "finetune – no rule" baseline:
  - Starts from the pretrained AR checkpoint (trained on A = {0..5})
  - Continues training on B = {2,3,4,5,6,7}
  - Saved to checkpoints/ar_finetuned.pt

Behaviour expected after fine-tuning:
  - Good on B (finetune-only {6,7} and both {2-5})
  - May forget pretrain-only starters {0,1} (catastrophic forgetting)
  - Still fails on neither starters {8-11} (no generalisation signal)
"""

from __future__ import annotations

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset    import (get_dataloaders, AR_TRAIN_STARTERS, FINETUNE_STARTERS,
                              TEST_STARTERS, VOCAB_SIZE)
from models.transformer import AutoregressiveTransformer


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pretrain set A : {AR_TRAIN_STARTERS}")
    print(f"Finetune set B : {FINETUNE_STARTERS}")
    print(f"Test starters  : {TEST_STARTERS}")

    train_loader, test_loader = get_dataloaders(
        batch_size          = args.batch_size,
        n_cycles            = args.n_cycles,
        n_seqs_per_starter  = args.n_seqs_per_starter,
        train_starters      = FINETUNE_STARTERS,
    )

    model = AutoregressiveTransformer(
        vocab_size  = VOCAB_SIZE,
        max_seq_len = args.n_cycles * 4 + 10,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
    ).to(device)

    # Load pretrained weights
    if args.ar_ckpt and os.path.exists(args.ar_ckpt):
        model.load_state_dict(torch.load(args.ar_ckpt, map_location=device))
        print(f"Loaded pretrained AR weights from {args.ar_ckpt}")
    else:
        print("WARNING: pretrained checkpoint not found; fine-tuning from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_train_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss    += loss.item() * tgt.numel()
            total_correct += (logits.argmax(-1) == tgt).sum().item()
            total_tokens  += tgt.numel()

        scheduler.step()
        train_loss = total_loss / total_tokens
        train_acc  = total_correct / total_tokens

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
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"test_acc(unseen)={test_acc:.4f}"
            )

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            path = os.path.join(args.ckpt_dir, "ar_finetuned.pt")
            torch.save(model.state_dict(), path)

    print(f"\nSaved fine-tuned AR checkpoint → {os.path.join(args.ckpt_dir, 'ar_finetuned.pt')}")
    print(f"Best train accuracy on finetune set: {best_train_acc:.4f}")


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune AR transformer on FINETUNE_STARTERS")
    p.add_argument("--ar_ckpt",            type=str,   default="checkpoints/ar_transformer.pt",
                   help="Pretrained AR checkpoint to start from")
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=1e-4,
                   help="Lower LR than pretraining to avoid forgetting")
    p.add_argument("--d_model",            type=int,   default=128)
    p.add_argument("--n_layers",           type=int,   default=4)
    p.add_argument("--n_heads",            type=int,   default=4)
    p.add_argument("--n_cycles",           type=int,   default=8)
    p.add_argument("--n_seqs_per_starter", type=int,   default=200)
    p.add_argument("--log_every",          type=int,   default=20)
    p.add_argument("--ckpt_dir",           type=str,   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
