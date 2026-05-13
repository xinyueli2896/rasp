"""
Fine-tune a pretrained CP bass transformer on new keys.

Loads *weights only* from a pretrain checkpoint (no optimizer / scheduler
state), then trains with a lower cosine-decay LR so the model learns the
finetune-new keys without catastrophic forgetting of pretrain keys.

Usage
-----
  # 1. Generate finetune data (keys 1, 3, 10 — not seen during pretrain)
  python -m midi_adapter.generate_synthetic_bass \\
      --n_songs 1500 --n_bars 128 \\
      --out_dir data/bass_finetune --out_pt data/bass_finetune_cp4 \\
      --keys 1 3 10

  # 2. Fine-tune from the pretrain checkpoint
  python -m midi_adapter.finetune_cp_bass \\
      --pretrain_ckpt ckpt/cp_bass_size1_batch8/last.ckpt \\
      --train_data    data/bass_finetune_cp4.pt \\
      --val_data      data/bass_pretrain_cp4.pt \\
      --model_size 1 --batch_size 8

Key differences vs pretrain_cp_bass.py
---------------------------------------
  * --pretrain_ckpt  : loads weights only (fresh optimizer)
  * max_lr defaults to 2e-5 (5× lower than pretrain)
  * LR schedule      : cosine decay, short 1% warmup (no spike)
  * max_steps        : 50 000 (default)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer, FramedDataset

TRAIN_LENGTH  = 384
MAX_STEPS     = 50_000


def _load_weights(net: RoFormerSymbolicTransformer, ckpt_path: str) -> None:
    """Load weights from a .pt or .ckpt file, ignoring optimizer state."""
    state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state:      # Lightning checkpoint
        state = state['state_dict']
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'  WARNING missing keys  : {missing[:5]}{"..." if len(missing)>5 else ""}')
    if unexpected:
        print(f'  WARNING unexpected    : {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
    print(f'  Weights loaded from   : {ckpt_path}')


def main(args):
    n_gpus   = max(torch.cuda.device_count(), 1)
    run_name = (
        args.run_name
        or f'cp_bass_ft_size{args.model_size}_batch{args.batch_size * n_gpus}'
    )

    pretrain_max_lr = 5e-5 if args.model_size >= 2 else 1e-4
    net = RoFormerSymbolicTransformer(
        size         = args.model_size,
        max_lr       = pretrain_max_lr,   # used internally; we override the optimizer
        with_velocity= False,
    )

    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        _load_weights(net, args.pretrain_ckpt)
    else:
        print(f'  WARNING: --pretrain_ckpt not found ({args.pretrain_ckpt}) — random init')

    ft_lr = args.ft_lr
    def _configure_optimizers(self=net):
        optimizer = torch.optim.AdamW(net.parameters(), lr=ft_lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=ft_lr,
            total_steps=args.max_steps,
            pct_start=0.01,       # 1% warmup → shorter spike than pretrain
            anneal_strategy='cos',
        )
        return [optimizer], [scheduler]
    net.configure_optimizers = _configure_optimizers

    train_loader = DataLoader(
        FramedDataset(args.train_data, TRAIN_LENGTH, args.batch_size,
                      split=args.train_split),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_loader = DataLoader(
        FramedDataset(args.val_data, TRAIN_LENGTH, args.batch_size,
                      split=args.val_split, sample_step=16, repeat=True),
        batch_size=None, num_workers=0,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_cb = L.callbacks.ModelCheckpoint(
        monitor    = 'val_loss',
        save_top_k = 5,
        save_last  = True,
        enable_version_counter = False,
        dirpath    = os.path.join(args.ckpt_dir, run_name),
        filename   = run_name + '.{epoch:02d}.{val_loss:.5f}',
    )

    precision = 'bf16-mixed' if torch.cuda.is_available() else 32

    if n_gpus > 1:
        import datetime
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'

    loggers = []
    if args.wandb_project:
        loggers.append(WandbLogger(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            config  = vars(args),
        ))
    loggers.append(TensorBoardLogger('tb_logs', name=run_name))

    use_gpu = torch.cuda.is_available()
    trainer = L.Trainer(
        devices            = -1 if use_gpu else 1,
        accelerator        = 'gpu' if use_gpu else 'cpu',
        precision          = precision,
        max_steps          = args.max_steps,
        callbacks          = [checkpoint_cb],
        val_check_interval = args.val_check_interval,
        limit_val_batches  = 200,
        check_val_every_n_epoch = None,
        gradient_clip_val  = 1.0,
        logger             = loggers,
        num_sanity_val_steps = 2,
        strategy           = strategy,
    )

    trainer.fit(net, train_loader, val_loader)

    out_pt = os.path.join(args.ckpt_dir, f'{run_name}.pt')
    torch.save(net.state_dict(), out_pt)
    print(f'Model saved → {out_pt}')


def get_args():
    p = argparse.ArgumentParser(
        description='Fine-tune CP transformer on new keys from a pretrain checkpoint'
    )
    p.add_argument('--pretrain_ckpt', required=True,
                   help='.pt or .ckpt file with pretrain weights (optimizer state ignored)')
    p.add_argument('--train_data',    required=True,
                   help='Finetune training .pt file (e.g. data/bass_finetune_cp4.pt)')
    p.add_argument('--val_data',      required=True,
                   help='Validation .pt file (use pretrain data to monitor forgetting)')
    p.add_argument('--train_split',  type=str, default='train',
                   choices=['all', 'train', 'val', 'test'])
    p.add_argument('--val_split',    type=str, default='val',
                   choices=['all', 'train', 'val', 'test'])
    p.add_argument('--model_size',   type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--ft_lr',        type=float, default=2e-5,
                   help='Peak LR for fine-tuning (default 2e-5, ~5× lower than pretrain)')
    p.add_argument('--max_steps',          type=int, default=MAX_STEPS)
    p.add_argument('--val_check_interval', type=int, default=1000)
    p.add_argument('--ckpt_dir',      type=str, default='ckpt')
    p.add_argument('--run_name',      type=str, default=None)
    p.add_argument('--wandb_project', type=str, default='cp_bass',
                   help='W&B project name (empty string to disable wandb)')
    p.add_argument('--wandb_entity',  type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
