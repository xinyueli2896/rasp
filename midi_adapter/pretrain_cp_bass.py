"""
Pretrain RoFormerSymbolicTransformer on synthetic bass data.

Usage
-----
  python -m midi_adapter.pretrain_cp_bass \\
      --train_data data/bass_pretrain_cp4.pt \\
      --val_data   data/bass_all_cp4.pt \\
      --model_size 1 --batch_size 8

The script is a thin wrapper around cp_transformer.py that accepts data paths
as arguments instead of hardcoding them.
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

TRAIN_LENGTH = 24
MAX_STEPS    = 200_000


def main(args):
    n_gpus     = max(torch.cuda.device_count(), 1)
    max_lr     = 5e-5 if args.model_size >= 2 else 1e-4
    run_name   = (
        args.run_name
        or f'cp_bass_size{args.model_size}_batch{args.batch_size * n_gpus}'
    )

    net = RoFormerSymbolicTransformer(
        size         = args.model_size,
        max_lr       = max_lr,
        with_velocity= False,
    )

    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        net.load_state_dict(state)
        print(f'Loaded pretrained weights from {args.base_ckpt}')

    # cp_transformer's configure_optimizers hardcodes total_steps=2_000_000.
    # Override to use the actual training budget so the LR schedule is correct.
    # Return legacy ([optimizer], [scheduler]) so PL steps per epoch (never, since we
    # use max_steps), and training_step's manual scheduler.step() handles per-step LR.
    def _configure_optimizers(self=net):
        optimizer = torch.optim.AdamW(net.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=args.max_steps, pct_start=0.005,
        )
        return [optimizer], [scheduler]
    net.configure_optimizers = _configure_optimizers

    def _preload(ds, cache: dict) -> None:
        """Pre-load data into ds so the worker process never makes a second copy."""
        path = ds.file_path
        if path not in cache:
            cache[path] = torch.load(path, weights_only=True)
            print(f'Pre-loaded {path}')
        ds.data = cache[path]
        psr = torch.load(path[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
        psr[psr[:, 0] < -5, 0] = -5
        psr[psr[:, 1] > 6, 1] = 6
        if ds.split in ('val', 'test'):
            psr = torch.zeros_like(psr)
        ds.pitch_shift_range = psr

    _cache: dict = {}
    train_ds = FramedDataset(args.train_data, TRAIN_LENGTH, args.batch_size,
                             split=args.train_split)
    _preload(train_ds, _cache)

    val_ds = FramedDataset(args.val_data, TRAIN_LENGTH, args.batch_size,
                           split=args.val_split, sample_step=16, repeat=True)
    _preload(val_ds, _cache)

    # num_workers=0: keeps both datasets in the main process so _cache sharing works.
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=0)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_cb = L.callbacks.ModelCheckpoint(
        monitor    = 'val_loss',
        save_top_k = 5,
        save_last  = True,
        save_weights_only = True,   # omit optimizer states: 1.1 GB → ~220 MB per ckpt
        enable_version_counter = False,
        dirpath    = os.path.join(args.ckpt_dir, run_name),
        filename   = run_name + '.{epoch:02d}.{val_loss:.5f}',
    )

    gradient_clip = 1.0
    precision     = 'bf16-mixed' if torch.cuda.is_available() else 32

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
        gradient_clip_val  = gradient_clip,
        logger             = loggers,
        num_sanity_val_steps = 2,
        strategy           = strategy,
    )

    ckpt_path = args.resume_ckpt if args.resume_ckpt and os.path.exists(args.resume_ckpt) else None
    trainer.fit(net, train_loader, val_loader, ckpt_path=ckpt_path)

    out_pt = os.path.join(args.ckpt_dir, f'{run_name}.pt')
    torch.save(net.state_dict(), out_pt)
    print(f'Model saved → {out_pt}')


def get_args():
    p = argparse.ArgumentParser(description='Pretrain CP transformer on synthetic bass')
    p.add_argument('--train_data',   required=True, help='Path to training .pt file')
    p.add_argument('--val_data',     required=True, help='Path to validation .pt file')
    p.add_argument('--train_split',  type=str, default='train',
                   choices=['all', 'train', 'val', 'test'],
                   help='FramedDataset split for train loader (default: train)')
    p.add_argument('--val_split',    type=str, default='val',
                   choices=['all', 'train', 'val', 'test'],
                   help='FramedDataset split for val loader (default: val)')
    p.add_argument('--model_size',  type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--max_steps',          type=int, default=MAX_STEPS)
    p.add_argument('--val_check_interval', type=int, default=2500)
    p.add_argument('--ckpt_dir',      type=str,  default='ckpt')
    p.add_argument('--run_name',      type=str,  default=None)
    p.add_argument('--base_ckpt',      type=str,  default=None,
                   help='Load pretrained weights before finetuning (starts with fresh optimizer/schedule)')
    p.add_argument('--resume_ckpt',   type=str,  default=None)
    p.add_argument('--wandb_project', type=str,  default='cp_bass',
                   help='W&B project name (set to empty string to disable wandb)')
    p.add_argument('--wandb_entity',  type=str,  default=None)
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
