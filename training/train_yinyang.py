
from __future__ import annotations

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset  import (get_adapter_dataloaders,
                            AR_TRAIN_STARTERS, ADAPTER_TRAIN_STARTERS,
                            TEST_STARTERS, VOCAB_SIZE)
from models.yinyang_model            import build_yinyang_model, RULE_D_MODEL


def _run_training(model, train_loader, test_loader, args, device, epochs, ckpt_path):
    """Core training loop — shared across phases."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Trainable: {n_trainable:,}   Frozen: {n_frozen:,}')

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
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

        model.eval()
        with torch.no_grad():
            test_correct, test_tokens = 0, 0
            for inp, tgt in test_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)
                test_correct += (logits.argmax(-1) == tgt).sum().item()
                test_tokens  += tgt.numel()
        test_acc = test_correct / test_tokens

        if epoch % args.log_every == 0 or epoch == epochs:
            print(
                f'Epoch {epoch:4d}/{epochs}  '
                f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
                f'test_acc(unseen)={test_acc:.4f}'
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            _save(model, ckpt_path, args.use_lora)

    print(f'Best test accuracy on unseen starters: {best_test_acc:.4f}')
    print(f'Saved checkpoint → {ckpt_path}')
    return best_test_acc


def _save(model, path, has_lora):
    state = {k: v for k, v in model.state_dict().items()
             if k.startswith('yinyang_attn')}
    if has_lora:
        for k, v in model.ar_model.state_dict().items():
            if 'lora_' in k:
                state[f'ar_model.{k}'] = v
    torch.save(state, path)


def _load_yinyang_attn(model, path):
    """Load only yinyang_attn weights from a checkpoint (ignores LoRA / AR weights)."""
    saved = torch.load(path, map_location='cpu')
    attn_state = {k.removeprefix('yinyang_attn.'): v
                  for k, v in saved.items() if k.startswith('yinyang_attn.')}
    model.yinyang_attn.load_state_dict(attn_state)
    print(f'Loaded yinyang_attn weights from {path}')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def train(args):
    set_seed(args.seed)
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
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f'{args.ckpt_name}.pt')

    def _build(use_lora):
        return build_yinyang_model(
            ar_ckpt_path   = args.ar_ckpt,
            max_seq_len    = args.n_cycles * 4 + 10,
            d_model        = args.d_model,
            n_layers       = args.n_layers,
            n_heads        = args.n_heads,
            rule_d_model   = RULE_D_MODEL,   # 28 — fixed by Tracr architecture
            adapter_rank   = args.adapter_rank,
            n_skip         = args.n_skip,
            use_lora       = use_lora,
            lora_rank      = args.lora_rank,
            force_fallback = args.force_fallback,
            device         = str(device),
        )

    if args.phase2_epochs > 0:
        # ------------------------------------------------------------------ #
        # Option 3: Two-phase training
        # Phase 1: AR frozen, only yinyang_attn trains
        # Phase 2: add LoRA to AR, warm-start yinyang_attn from phase 1
        # ------------------------------------------------------------------ #
        phase1_path = os.path.join(args.ckpt_dir, f'{args.ckpt_name}_phase1.pt')

        print(f'\n=== Phase 1 (no LoRA, {args.epochs} epochs) ===')
        model = _build(use_lora=False)
        print(f'yinyang_attn modules: {len(model.yinyang_attn)}  (every {args.n_skip} layers)')
        _run_training(model, train_loader, test_loader, args, device,
                      epochs=args.epochs, ckpt_path=phase1_path)

        print(f'\n=== Phase 2 (LoRA rank={args.lora_rank}, {args.phase2_epochs} epochs) ===')
        model = _build(use_lora=True)
        _load_yinyang_attn(model, phase1_path)   # warm-start cross-attention from phase 1
        _run_training(model, train_loader, test_loader, args, device,
                      epochs=args.phase2_epochs, ckpt_path=ckpt_path)

    else:
        # ------------------------------------------------------------------ #
        # Option 1 (--no_lora) or Option 2 (--use_lora --lora_rank N)
        # ------------------------------------------------------------------ #
        model = _build(use_lora=args.use_lora)
        print(f'yinyang_attn modules: {len(model.yinyang_attn)}  (every {args.n_skip} layers)')
        lora_tag = f'LoRA rank={args.lora_rank}' if args.use_lora else 'no LoRA'
        print(f'Mode: {lora_tag}')
        _run_training(model, train_loader, test_loader, args, device,
                      epochs=args.epochs, ckpt_path=ckpt_path)


def get_args():
    p = argparse.ArgumentParser(description='Train Yin-Yang model')
    p.add_argument('--ar_ckpt',            type=str,   default='checkpoints/ar_transformer.pt')
    p.add_argument('--epochs',             type=int,   default=100)
    p.add_argument('--phase2_epochs',      type=int,   default=0,
                   help='If > 0: two-phase training. Phase 1 = no_lora for --epochs, '
                        'Phase 2 = LoRA for --phase2_epochs (warm-started from phase 1).')
    p.add_argument('--batch_size',         type=int,   default=64)
    p.add_argument('--lr',                 type=float, default=1e-4)
    p.add_argument('--d_model',            type=int,   default=128)
    p.add_argument('--n_layers',           type=int,   default=4)
    p.add_argument('--n_heads',            type=int,   default=4)
    p.add_argument('--adapter_rank',       type=int,   default=32)
    p.add_argument('--lora_rank',          type=int,   default=16,
                   help='LoRA rank for AR model (options 2 and 3)')
    p.add_argument('--n_skip',             type=int,   default=2)
    p.add_argument('--n_cycles',           type=int,   default=8)
    p.add_argument('--n_seqs_per_starter', type=int,   default=200)
    p.add_argument('--log_every',          type=int,   default=10)
    p.add_argument('--ckpt_dir',           type=str,   default='checkpoints')
    p.add_argument('--ckpt_name',          type=str,   default='yinyang',
                   help='Checkpoint filename stem (saved as <ckpt_dir>/<ckpt_name>.pt)')
    p.add_argument('--use_lora',           action='store_true', default=True)
    p.add_argument('--no_lora',            action='store_false', dest='use_lora')
    p.add_argument('--force_fallback',     action='store_true')
    p.add_argument('--seed',               type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(get_args())
