"""
Fine-tune the Yin-Yang chord adapter on a frozen pretrained CP transformer.

Only ChordRuleModel + yinyang_attn weights are trained.
The base RoFormerSymbolicTransformer is completely frozen.

Typical workflow
----------------
1. Preprocess chord sidecar file (once):
     python -m midi_adapter.preprocess_chords \\
         --txt  data/rwc_cp16_v2.txt \\
         --root /path/to/rwc/midi \\
         --length data/rwc_cp16_v2.length.pt \\
         --out  data/rwc_cp16_v2.bar_chords.pt

2. Train adapter:
     python -m midi_adapter.train_yinyang \\
         --base_ckpt  ckpt/cp_transformer_v0.42_size1.pt \\
         --model_size 1 \\
         --train_midi  data/rwc_cp16_v2.pt \\
         --train_chord data/rwc_cp16_v2.bar_chords.pt \\
         --val_midi    data/rwc_cp16_v2.pt \\
         --val_chord   data/rwc_cp16_v2.bar_chords.pt \\
         --batch_size 8

Checkpoints are saved to ckpt/<run_name>/ via Lightning ModelCheckpoint.
The final adapter-only state dict is also saved as ckpt/<run_name>.adapter.pt
(contains only chord_rule_model and yinyang_attn weights — much smaller than
the full checkpoint, and can be loaded on top of any compatible base model).
"""

from __future__ import annotations

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

# Make sure cp_transformer and midi_adapter are importable when run from the
# midi_yinyang repo root.  Adjust this path if the layout differs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.chord_dataset import ChordConditionedDataset

TRAIN_LENGTH = 384
MAX_STEPS    = 200_000


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class CPYinyangModule(L.LightningModule):

    def __init__(
        self,
        base_model:   RoFormerSymbolicTransformer,
        rule_d_model: int   = 128,
        adapter_rank: int   = 256,
        n_skip:       int   = 4,
        lr:           float = 1e-4,
        max_steps:    int   = MAX_STEPS,
    ):
        super().__init__()
        self.model     = CPYinyangTransformer(base_model, rule_d_model, adapter_rank, n_skip)
        self.lr        = lr
        self.max_steps = max_steps

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_frozen    = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f'[CPYinyangModule] trainable={n_trainable:,}  frozen={n_frozen:,}')

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, pitch_shift, chord_tokens = batch
        loss = self.model.loss(x, pitch_shift, chord_tokens)
        self.log('train_loss', loss, prog_bar=True)
        self.lr_schedulers().step()
        self.log('lr', self.lr_schedulers().get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, pitch_shift, chord_tokens = batch
        loss = self.model.loss(x, pitch_shift, chord_tokens)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr,
            total_steps=self.max_steps, pct_start=0.01,
        )
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------
    def save_adapter(self, path: str):
        """Save only the trainable adapter weights (chord_rule_model + yinyang_attn)."""
        adapter_state = {
            k: v for k, v in self.model.state_dict().items()
            if k.startswith('chord_rule_model') or k.startswith('yinyang_attn')
        }
        torch.save(adapter_state, path)
        print(f'Adapter weights saved → {path}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_model(ckpt_path: str, model_size: int, with_velocity: bool = False,
                    max_position_embeddings: int = 1536) -> RoFormerSymbolicTransformer:
    net = RoFormerSymbolicTransformer(
        size=model_size,
        max_position_embeddings=max_position_embeddings,
        with_velocity=with_velocity,
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # support both raw state-dict and Lightning checkpoint
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    net.load_state_dict(state)
    print(f'[load_base_model] loaded {ckpt_path}')
    return net


def make_loader(midi_path, chord_path, batch_size, train_length, split,
                sample_step=1, repeat=True) -> DataLoader:
    ds = ChordConditionedDataset(
        midi_path     = midi_path,
        chord_path    = chord_path,
        target_length = train_length,
        batch_size    = batch_size,
        split         = split,
        sample_step   = sample_step,
        repeat        = repeat,
    )
    return DataLoader(ds, batch_size=None, num_workers=1, persistent_workers=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    base_model = load_base_model(
        args.base_ckpt, args.model_size,
        with_velocity=args.with_velocity,
    )

    module = CPYinyangModule(
        base_model   = base_model,
        rule_d_model = args.rule_d_model,
        adapter_rank = args.adapter_rank,
        n_skip       = args.n_skip,
        lr           = args.lr,
        max_steps    = args.max_steps,
    )

    train_loader = make_loader(
        args.train_midi, args.train_chord,
        args.batch_size, args.train_length,
        split='train' if args.val_midi == args.train_midi else 'all',
    )
    val_loader = make_loader(
        args.val_midi, args.val_chord,
        args.batch_size, args.train_length,
        split='val' if args.val_midi == args.train_midi else 'all',
        sample_step=16, repeat=False,
    )

    run_name = (
        args.run_name
        or f'cp_yinyang_size{args.model_size}'
           f'_ruleD{args.rule_d_model}'
           f'_rank{args.adapter_rank}'
           f'_skip{args.n_skip}'
    )

    checkpoint_cb = L.callbacks.ModelCheckpoint(
        monitor    = 'val_loss',
        save_top_k = 5,
        save_last  = True,
        dirpath    = os.path.join(args.ckpt_dir, run_name),
        filename   = run_name + '.{epoch:02d}.{val_loss:.5f}',
        enable_version_counter=False,
    )

    n_gpus    = max(torch.cuda.device_count(), 1)
    precision = 'bf16-mixed' if torch.cuda.is_available() else 32

    trainer = L.Trainer(
        devices            = -1,
        accelerator        = 'gpu' if torch.cuda.is_available() else 'cpu',
        precision          = precision,
        max_steps          = args.max_steps,
        callbacks          = [checkpoint_cb],
        val_check_interval = 2500,
        limit_val_batches  = 25,
        check_val_every_n_epoch = None,
        gradient_clip_val  = 1.0,
        logger             = TensorBoardLogger('tb_logs', name=run_name),
        num_sanity_val_steps = 2,
        strategy           = 'auto' if n_gpus <= 1 else 'ddp',
    )

    ckpt_path = args.resume_ckpt if args.resume_ckpt and os.path.exists(args.resume_ckpt) else None
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)

    # save compact adapter-only weights
    adapter_out = os.path.join(args.ckpt_dir, f'{run_name}.adapter.pt')
    module.save_adapter(adapter_out)


def get_args():
    p = argparse.ArgumentParser(description='Train CP-Transformer Yin-Yang chord adapter')

    # model
    p.add_argument('--base_ckpt',    required=True, help='Pretrained CP transformer .pt checkpoint')
    p.add_argument('--model_size',   type=int, default=1, choices=[0,1,2,3])
    p.add_argument('--with_velocity',action='store_true', default=False)
    p.add_argument('--rule_d_model', type=int, default=128)
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip',       type=int, default=4,
                   help='Inject adapter every n_skip global transformer layers')

    # data
    p.add_argument('--train_midi',   required=True)
    p.add_argument('--train_chord',  required=True)
    p.add_argument('--val_midi',     required=True)
    p.add_argument('--val_chord',    required=True)
    p.add_argument('--train_length', type=int, default=TRAIN_LENGTH)

    # training
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--max_steps',    type=int, default=MAX_STEPS)
    p.add_argument('--resume_ckpt',  type=str, default=None)

    # output
    p.add_argument('--ckpt_dir',     type=str, default='ckpt')
    p.add_argument('--run_name',     type=str, default=None)

    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
