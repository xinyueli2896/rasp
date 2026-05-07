"""
Fine-tune CPYinyangTransformer on synthetic bass data with chord conditioning.

The base CP transformer is frozen; only ChordRuleModel + yinyang_attn adapters
are trained.  Chord tokens are loaded from the .bar_chords.pt file produced by
generate_synthetic_bass.py.

Usage
-----
  python -m midi_adapter.train_cp_yinyang \\
      --base_ckpt  checkpoints/cp_bass_size1_pretrain.pt \\
      --train_data data/bass_pretrain_cp4.pt \\
      --val_data   data/bass_pretrain_cp4.pt \\
      --train_split train --val_split val
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer, FramedDataset
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.chord_tokenizer import N_QUALITIES, NO_CHORD_TOKEN

TRAIN_LENGTH     = 384
SUBBEATS_PER_BAR = 16
MAX_STEPS        = 100_000


# ---------------------------------------------------------------------------
# Dataset — extends FramedDataset to also yield bar-level chord tokens
# ---------------------------------------------------------------------------

class ChordFramedDataset(FramedDataset):
    """
    Yields (data_window, pitch_shift, chord_tokens) where chord_tokens is a
    (B, n_bars) int64 tensor aligned to the sampled subbeat window.
    """

    def __init__(self, file_path, target_length, batch_size, **kwargs):
        super().__init__(file_path, target_length, batch_size, **kwargs)
        self.bar_chords_data = None
        self.n_bars = target_length // SUBBEATS_PER_BAR

    def __iter__(self):
        # Load main data (via parent lazy load)
        if self.data is None:
            self.data = torch.load(self.file_path, weights_only=True)
            self.pitch_shift_range = torch.load(
                self.file_path[:-3] + '.pitch_shift_range.pt', weights_only=True
            ).reshape(-1, 2)
            self.pitch_shift_range[self.pitch_shift_range[:, 0] < -5, 0] = -5
            self.pitch_shift_range[self.pitch_shift_range[:, 1] > 6, 1] = 6
            if self.split in ('val', 'test'):
                self.pitch_shift_range = torch.zeros_like(self.pitch_shift_range)
            print(f'Data for dataset {self.file_path} loaded.')

        if self.bar_chords_data is None:
            self.bar_chords_data = torch.load(
                self.file_path[:-3] + '.bar_chords.pt', weights_only=False
            )
            print(f'Bar chords for {self.file_path} loaded.')

        while True:
            if self.random_order:
                indices = torch.randperm(len(self.valid_indices))
            else:
                indices = torch.arange(len(self.valid_indices))

            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices     = indices[i:i + self.batch_size]
                raw_ids           = self.valid_indices[batch_indices]
                ps_range          = self.pitch_shift_range[raw_ids]

                starts = (
                    torch.floor(
                        torch.rand(len(raw_ids))
                        * (self.length[raw_ids] - self.target_length) / self.sample_step
                    ).long() * self.sample_step
                    + self.start[raw_ids]
                )
                index_matrix = (
                    torch.arange(self.target_length).view(1, -1) + starts.view(-1, 1)
                )
                pitch_shift = (
                    torch.floor(
                        torch.rand(len(raw_ids))
                        * (ps_range[:, 1] - ps_range[:, 0] + 1)
                    ).long() + ps_range[:, 0]
                )

                # Chord tokens: extract n_bars bars starting at the window's bar offset
                starts_in_song = starts - self.start[raw_ids]   # subbeat offset within song
                bar_starts     = (starts_in_song // SUBBEATS_PER_BAR).tolist()
                chord_list = []
                for song_idx, bar_start in zip(raw_ids.tolist(), bar_starts):
                    ct = self.bar_chords_data[song_idx]          # int16 tensor (n_bars_song,)
                    ct = ct[bar_start: bar_start + self.n_bars].long()
                    # pad if window overruns (shouldn't happen if song is long enough)
                    if ct.shape[0] < self.n_bars:
                        pad = torch.full((self.n_bars - ct.shape[0],), 420, dtype=torch.long)
                        ct  = torch.cat([ct, pad])
                    chord_list.append(ct)
                chord_tokens = torch.stack(chord_list, dim=0)    # (B, n_bars)

                yield self.data[index_matrix], pitch_shift, chord_tokens

            if not self.repeat:
                break


# ---------------------------------------------------------------------------
# Unseen accuracy callback
# ---------------------------------------------------------------------------

class UnseenAccuracyCallback(L.Callback):
    """
    At each val check, compute chord root accuracy on unseen-key data.

    Accuracy = fraction of bar-start positions where the model's predicted
    bass pitch (argmax of pitch-slot logits % 128) matches the expected
    MIDI pitch (36 + chord_root).

    Requires sample_step=16 in the dataloader so windows start at bar
    boundaries and chord_tokens[b, k] aligns with subbeat k*16.
    """

    def __init__(self, dataloader: DataLoader, n_batches: int = 25):
        self.dataloader = dataloader
        self.n_batches  = n_batches

    def on_validation_epoch_end(self, trainer, pl_module):
        model  = pl_module.model
        device = pl_module.device
        base   = model.base

        correct = 0
        total   = 0

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= self.n_batches:
                    break
                x, pitch_shift, chord_tokens = [t.to(device) for t in batch]
                B, seq_len, _ = x.shape

                x_proc = base.preprocess(x, pitch_shift)          # (B, seq_len, 8)
                logits  = model(x_proc, chord_tokens)              # (B*seq_len, 8, V)
                logits  = logits.view(B, seq_len, 8, -1)          # (B, seq_len, 8, V)

                # Slot 1 at bar-start subbeats (0, 16, 32, ...) predicts pitch encoding
                pitch_logits = logits[:, ::16, 1, :]               # (B, n_bars, V)
                pred_enc     = pitch_logits.argmax(-1)             # (B, n_bars)
                pred_pitch   = pred_enc % 128                      # decode MIDI pitch

                expected_root  = chord_tokens // N_QUALITIES       # (B, n_bars)
                expected_pitch = 36 + expected_root                # MIDI pitch C2=36..B2=47

                valid   = chord_tokens != NO_CHORD_TOKEN
                correct += (pred_pitch == expected_pitch)[valid].sum().item()
                total   += valid.sum().item()

        acc = correct / max(total, 1)
        pl_module.log('unseen_acc', acc, prog_bar=True)


# ---------------------------------------------------------------------------
# Lightning module wrapper
# ---------------------------------------------------------------------------

class CPYinyangLightning(L.LightningModule):

    def __init__(self, model: CPYinyangTransformer, max_lr: float, max_steps: int):
        super().__init__()
        self.model    = model
        self.max_lr   = max_lr
        self.max_steps = max_steps

    def forward(self, x, chord_tokens):
        return self.model(x, chord_tokens)

    def training_step(self, batch, batch_idx):
        x, pitch_shift, chord_tokens = batch
        loss = self.model.loss(x, pitch_shift, chord_tokens)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
            self.log('training/lr', scheduler.get_last_lr()[0], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, pitch_shift, chord_tokens = batch
        loss = self.model.loss(x, pitch_shift, chord_tokens)
        key  = 'val_loss' if dataloader_idx == 0 else 'unseen_loss'
        self.log(key, loss, on_step=False, on_epoch=True,
                 sync_dist=True, add_dataloader_idx=False)
        return loss

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.max_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr,
            total_steps=self.max_steps, pct_start=0.02,
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    n_gpus   = max(torch.cuda.device_count(), 1)
    max_lr   = 5e-5 if args.model_size >= 2 else 1e-4
    run_name = (
        args.run_name
        or f'cp_yinyang_size{args.model_size}_rank{args.adapter_rank}'
    )

    # Build base CP transformer and load pretrained weights
    base = RoFormerSymbolicTransformer(
        size=args.model_size, max_lr=max_lr, with_velocity=False,
    )
    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location='cpu')
        base.load_state_dict(state)
        print(f'Loaded base CP transformer from {args.base_ckpt}')
    else:
        print('WARNING: no base checkpoint found — training adapter from scratch.')

    adapter = CPYinyangTransformer(
        base_model   = base,
        adapter_rank = args.adapter_rank,
        n_skip       = args.n_skip,
    )

    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in adapter.parameters() if not p.requires_grad)
    print(f'Trainable: {n_trainable:,}   Frozen: {n_frozen:,}')

    lit = CPYinyangLightning(adapter, max_lr=max_lr, max_steps=args.max_steps)

    train_loader = DataLoader(
        ChordFramedDataset(args.train_data, TRAIN_LENGTH, args.batch_size,
                           split=args.train_split),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_loader = DataLoader(
        ChordFramedDataset(args.val_data, TRAIN_LENGTH, args.batch_size,
                           split=args.val_split, sample_step=16, repeat=True),
        batch_size=None, num_workers=0,
    )

    val_loaders   = [val_loader]
    unseen_acc_cb = None
    if args.unseen_data and os.path.exists(args.unseen_data):
        unseen_loader = DataLoader(
            ChordFramedDataset(args.unseen_data, TRAIN_LENGTH, args.batch_size,
                               split='all', sample_step=16, repeat=True),
            batch_size=None, num_workers=0,
        )
        val_loaders.append(unseen_loader)
        unseen_acc_cb = UnseenAccuracyCallback(unseen_loader, n_batches=25)
        print(f'Unseen eval data: {args.unseen_data}')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_cb = L.callbacks.ModelCheckpoint(
        monitor    = 'val_loss',
        save_top_k = 3,
        save_last  = True,
        enable_version_counter = False,
        dirpath    = os.path.join(args.ckpt_dir, run_name),
        filename   = run_name + '.{epoch:02d}.{val_loss:.5f}',
    )

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
    if n_gpus > 1:
        import datetime
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'

    callbacks = [checkpoint_cb]
    if unseen_acc_cb is not None:
        callbacks.append(unseen_acc_cb)

    trainer = L.Trainer(
        devices            = -1 if use_gpu else 1,
        accelerator        = 'gpu' if use_gpu else 'cpu',
        precision          = 'bf16-mixed' if use_gpu else 32,
        max_steps          = args.max_steps,
        callbacks          = callbacks,
        val_check_interval = args.val_check_interval,
        limit_val_batches  = 25,
        check_val_every_n_epoch = None,
        gradient_clip_val  = 1.0,
        logger             = loggers,
        num_sanity_val_steps = 2,
        strategy           = strategy,
    )

    ckpt_path = args.resume_ckpt if args.resume_ckpt and os.path.exists(args.resume_ckpt) else None
    trainer.fit(lit, train_loader, val_loaders, ckpt_path=ckpt_path)

    out_pt = os.path.join(args.ckpt_dir, f'{run_name}.pt')
    torch.save(adapter.state_dict(), out_pt)
    print(f'Adapter saved → {out_pt}')


def get_args():
    p = argparse.ArgumentParser(description='Train CPYinyangTransformer adapter')
    p.add_argument('--base_ckpt',          type=str, required=True,
                   help='Path to pretrained CP transformer .pt file')
    p.add_argument('--train_data',         type=str, required=True)
    p.add_argument('--val_data',           type=str, required=True)
    p.add_argument('--train_split',        type=str, default='train',
                   choices=['all', 'train', 'val', 'test'])
    p.add_argument('--val_split',          type=str, default='val',
                   choices=['all', 'train', 'val', 'test'])
    p.add_argument('--model_size',         type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--batch_size',         type=int, default=8)
    p.add_argument('--max_steps',          type=int, default=MAX_STEPS)
    p.add_argument('--val_check_interval', type=int, default=500)
    p.add_argument('--adapter_rank',       type=int, default=256)
    p.add_argument('--n_skip',             type=int, default=4)
    p.add_argument('--ckpt_dir',           type=str, default='checkpoints')
    p.add_argument('--run_name',           type=str, default=None)
    p.add_argument('--resume_ckpt',        type=str, default=None)
    p.add_argument('--unseen_data',        type=str, default=None,
                   help='Path to unseen-keys .pt file for generalisation eval')
    p.add_argument('--wandb_project',      type=str, default='cp_bass')
    p.add_argument('--wandb_entity',       type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
