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
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR

TRAIN_LENGTH = 24
MAX_STEPS    = 100_000


# ---------------------------------------------------------------------------
# Dataset — extends FramedDataset to also yield bar-level chord tokens
# ---------------------------------------------------------------------------

class ChordFramedDataset(FramedDataset):
    """
    Yields (data_window, pitch_shift, chord_tokens) where chord_tokens is a
    (B, n_beats) int64 tensor at beat granularity, aligned to the sampled window.
    """

    def __init__(self, file_path, target_length, batch_size,
                 disable_pitch_shift: bool = False, **kwargs):
        super().__init__(file_path, target_length, batch_size, **kwargs)
        self.beat_chords_data     = None
        self.n_beats              = target_length   # one chord token per beat
        self.disable_pitch_shift  = disable_pitch_shift

    def preload(self, shared_data_cache: dict | None = None):
        """Load data, pitch_shift_range, and beat_chords into memory now.
        Pass a shared_cache dict to reuse tensors across datasets backed by the same file.
        This avoids duplicate copies when train and val use the same .pt file.
        """
        cache = shared_data_cache if shared_data_cache is not None else {}
        path  = self.file_path

        if path not in cache:
            cache[path] = torch.load(path, weights_only=True)
            print(f'Pre-loaded {path}')
        self.data = cache[path]

        # pitch_shift_range is small; keep per-instance so split-specific zeroing is safe
        psr_path = path[:-3] + '.pitch_shift_range.pt'
        psr = torch.load(psr_path, weights_only=True).reshape(-1, 2)
        psr[psr[:, 0] < -5, 0] = -5
        psr[psr[:, 1] > 6, 1] = 6
        if self.split in ('val', 'test'):
            psr = torch.zeros_like(psr)
        self.pitch_shift_range = psr

        chords_key = path[:-3] + '.beat_chords.pt'
        if chords_key not in cache:
            cache[chords_key] = torch.load(chords_key, weights_only=False)
            print(f'Pre-loaded chords {chords_key}')
        self.beat_chords_data = cache[chords_key]

    def __iter__(self):
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

        if self.beat_chords_data is None:
            self.beat_chords_data = torch.load(
                self.file_path[:-3] + '.beat_chords.pt', weights_only=False
            )
            print(f'Beat chords for {self.file_path} loaded.')

        while True:
            if self.random_order:
                indices = torch.randperm(len(self.valid_indices))
            else:
                indices = torch.arange(len(self.valid_indices))

            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                raw_ids       = self.valid_indices[batch_indices]
                ps_range      = self.pitch_shift_range[raw_ids]

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
                if self.disable_pitch_shift:
                    pitch_shift = torch.zeros(len(raw_ids), dtype=torch.long)
                else:
                    pitch_shift = (
                        torch.floor(
                            torch.rand(len(raw_ids))
                            * (ps_range[:, 1] - ps_range[:, 0] + 1)
                        ).long() + ps_range[:, 0]
                    )

                # Beat-level chord tokens: one per beat, aligned to window start
                beat_starts = (starts - self.start[raw_ids]).tolist()
                chord_list  = []
                for song_idx, beat_start in zip(raw_ids.tolist(), beat_starts):
                    ct = self.beat_chords_data[song_idx]   # int16 (n_beats_song,)
                    ct = ct[beat_start: beat_start + self.n_beats].long()
                    if ct.shape[0] < self.n_beats:
                        pad = torch.full((self.n_beats - ct.shape[0],), NO_CHORD_TOKEN,
                                         dtype=torch.long)
                        ct  = torch.cat([ct, pad])
                    chord_list.append(ct)
                chord_tokens = torch.stack(chord_list, dim=0)   # (B, n_beats)

                yield self.data[index_matrix], pitch_shift, chord_tokens

            if not self.repeat:
                break


# ---------------------------------------------------------------------------
# Unseen accuracy callback
# ---------------------------------------------------------------------------

class UnseenAccuracyCallback(L.Callback):
    """
    At each val check, compute chord root accuracy on unseen-key data using
    AUTOREGRESSIVE generation so the metric reflects actual generation quality
    rather than teacher-forced next-token prediction.

    For each batch: preprocess the first n_prompt_beats as a prompt, generate
    the next n_gen_beats autoregressively (greedy), then check whether the
    pitch class of each generated beat matches the chord root.
    """

    def __init__(self, dataloader: DataLoader, n_batches: int = 5,
                 n_prompt_beats: int = 4, n_gen_beats: int = 16):
        self.dataloader     = dataloader
        self.n_batches      = n_batches
        self.n_prompt_beats = n_prompt_beats
        self.n_gen_beats    = n_gen_beats

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        # AR generation is expensive; only run on rank 0 to avoid DDP OOM / sync issues
        if trainer.global_rank != 0:
            return
        model       = pl_module.model
        device      = pl_module.device
        total_beats = self.n_prompt_beats + self.n_gen_beats

        correct = 0
        total   = 0

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= self.n_batches:
                    break
                x, pitch_shift, chord_tokens = [t.to(device) for t in batch]

                # Preprocess prompt (pitch_shift=0 so preprocess is a no-op on values)
                prompt    = model.base.preprocess(
                    x[:, :self.n_prompt_beats, :], pitch_shift
                )                                              # (B, n_prompt_beats, 8)
                chord_tok = chord_tokens[:, :total_beats]      # (B, total_beats)

                # Autoregressive generation — greedy (temperature=0)
                sampled = model.global_sampling_chord(
                    prompt, chord_tok,
                    max_seq_len=total_beats, temperature=0,
                )

                # sampled[t] is (B, subseq_len) preprocessed token for beat t.
                # Slot 1 = pitch + (dur+1)*128  →  token % 128 = pitch  →  % 12 = pc
                for t in range(self.n_prompt_beats, total_beats):
                    y_t         = sampled[t]                   # (B, 8)
                    pred_pc     = (y_t[:, 1] % 128) % 12      # (B,)
                    ct          = chord_tok[:, t]              # (B,)
                    valid       = ct != NO_CHORD_TOKEN
                    expected_pc = ct // N_QUALITIES            # (B,)
                    correct    += (pred_pc == expected_pc)[valid].sum().item()
                    total      += valid.sum().item()

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
        # Shift chord roots to match pitch-shifted notes so adapter sees consistent signal
        roots   = chord_tokens // N_QUALITIES
        quals   = chord_tokens % N_QUALITIES
        valid   = chord_tokens != NO_CHORD_TOKEN
        shifted = ((roots + pitch_shift[:, None]) % 12) * N_QUALITIES + quals
        chord_tokens = torch.where(valid, shifted, chord_tokens)
        loss = self.model.loss(x, pitch_shift, chord_tokens)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
            self.log('training/lr', scheduler.get_last_lr()[0], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, pitch_shift, chord_tokens = batch
        roots   = chord_tokens // N_QUALITIES
        quals   = chord_tokens % N_QUALITIES
        valid   = chord_tokens != NO_CHORD_TOKEN
        shifted = ((roots + pitch_shift[:, None]) % 12) * N_QUALITIES + quals
        chord_tokens = torch.where(valid, shifted, chord_tokens)
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
    lora_suffix = f'_lora{args.lora_rank}' if args.lora_rank > 0 else ''
    run_name = (
        args.run_name
        or f'cp_yinyang_size{args.model_size}_rank{args.adapter_rank}_skip{args.n_skip}{lora_suffix}'
    )

    # Build base CP transformer and load pretrained weights
    base = RoFormerSymbolicTransformer(
        size=args.model_size, max_lr=max_lr, with_velocity=False,
    )
    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location='cpu')
        if 'state_dict' in state:   # Lightning .ckpt
            state = state['state_dict']
        base.load_state_dict(state)
        print(f'Loaded base CP transformer from {args.base_ckpt}')
    else:
        print('WARNING: no base checkpoint found — training adapter from scratch.')

    adapter = CPYinyangTransformer(
        base_model   = base,
        adapter_rank = args.adapter_rank,
        n_skip       = args.n_skip,
        lora_rank    = args.lora_rank,
    )

    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in adapter.parameters() if not p.requires_grad)
    print(f'Trainable: {n_trainable:,}   Frozen: {n_frozen:,}')

    lit = CPYinyangLightning(adapter, max_lr=max_lr, max_steps=args.max_steps)

    # Shared cache so datasets pointing to the same file reuse one tensor copy
    _cache: dict = {}

    train_ds = ChordFramedDataset(args.train_data, TRAIN_LENGTH, args.batch_size,
                                  split=args.train_split, sample_step=SUBBEATS_PER_BAR,
                                  disable_pitch_shift=True)
    train_ds.preload(_cache)

    val_ds = ChordFramedDataset(args.val_data, TRAIN_LENGTH, args.batch_size,
                                split=args.val_split, sample_step=SUBBEATS_PER_BAR,
                                repeat=True, disable_pitch_shift=True)
    val_ds.preload(_cache)

    # num_workers=0: all data loading stays in the main process so _cache sharing works.
    # A worker subprocess would copy-on-write the tensor and defeat the sharing.
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=0)

    val_loaders   = [val_loader]
    unseen_acc_cb = None
    if args.unseen_data and os.path.exists(args.unseen_data):
        unseen_ds = ChordFramedDataset(args.unseen_data, TRAIN_LENGTH, args.batch_size,
                                       split='val', sample_step=SUBBEATS_PER_BAR,
                                       repeat=True, disable_pitch_shift=True)
        unseen_ds.preload(_cache)
        unseen_loader = DataLoader(unseen_ds, batch_size=None, num_workers=0)
        val_loaders.append(unseen_loader)
        unseen_acc_cb = UnseenAccuracyCallback(unseen_loader, n_batches=5,
                                                n_prompt_beats=4, n_gen_beats=16)
        print(f'Unseen eval data: {args.unseen_data}')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    if unseen_acc_cb is not None:
        # Mirror IS protocol: save best checkpoint by unseen-key accuracy
        checkpoint_cb = L.callbacks.ModelCheckpoint(
            monitor    = 'unseen_acc',
            mode       = 'max',
            save_top_k = 3,
            save_last  = True,
            enable_version_counter = False,
            dirpath    = os.path.join(args.ckpt_dir, run_name),
            filename   = run_name + '.{epoch:02d}.{unseen_acc:.4f}',
        )
    else:
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

    # Load best checkpoint and save adapter weights as plain .pt
    best_ckpt = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        best_state = torch.load(best_ckpt, map_location='cpu')['state_dict']
        adapter_state = {k[len('model.'):]: v for k, v in best_state.items()
                         if k.startswith('model.')}
        print(f'Best checkpoint: {best_ckpt}')
    else:
        adapter_state = adapter.state_dict()

    out_pt = os.path.join(args.ckpt_dir, f'{run_name}.pt')
    torch.save(adapter_state, out_pt)
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
    p.add_argument('--lora_rank',          type=int, default=0,
                   help='LoRA rank for base model Q/V projections (0 = frozen base)')
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
