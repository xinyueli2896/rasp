"""
Fine-tune CPYinyangTransformer on synthetic bass data with rule conditioning.

The base CP transformer is frozen; only yinyang_attn adapters are trained.
The rule model is analytical (BassTracrRuleModel) and reads the starting pitch
class directly from the generated sequence — no chord token files needed.

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
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS

TRAIN_LENGTH = 24
MAX_STEPS    = 100_000


# ---------------------------------------------------------------------------
# Dataset — extends FramedDataset to also yield bar-level chord tokens
# ---------------------------------------------------------------------------

class InterleavedDataset(IterableDataset):
    """Interleave two IterableDatasets, sampling each with probability p_first / (1-p_first)."""

    def __init__(self, ds1: 'BassFramedDataset', ds2: 'BassFramedDataset', p_first: float = 0.5):
        self.ds1      = ds1
        self.ds2      = ds2
        self.p_first  = p_first

    def __iter__(self):
        it1 = iter(self.ds1)
        it2 = iter(self.ds2)
        while True:
            yield next(it1) if torch.rand(1).item() < self.p_first else next(it2)


class BassFramedDataset(FramedDataset):
    """
    Thin extension of FramedDataset that adds shared-cache preloading and
    an optional disable_pitch_shift flag.  Yields (data_window, pitch_shift).
    """

    def __init__(self, file_path, target_length, batch_size,
                 disable_pitch_shift: bool = False, **kwargs):
        super().__init__(file_path, target_length, batch_size, **kwargs)
        self.disable_pitch_shift = disable_pitch_shift

    def preload(self, shared_data_cache: dict | None = None):
        """Load data and pitch_shift_range into memory now.
        Pass a shared_cache dict to reuse tensors across datasets backed by the same file.
        """
        cache = shared_data_cache if shared_data_cache is not None else {}
        path  = self.file_path

        if path not in cache:
            cache[path] = torch.load(path, weights_only=True)
            print(f'Pre-loaded {path}')
        self.data = cache[path]

        psr_path = path[:-3] + '.pitch_shift_range.pt'
        psr = torch.load(psr_path, weights_only=True).reshape(-1, 2)
        psr[psr[:, 0] < -5, 0] = -5
        psr[psr[:, 1] > 6, 1] = 6
        if self.split in ('val', 'test'):
            psr = torch.zeros_like(psr)
        self.pitch_shift_range = psr

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

                yield self.data[index_matrix], pitch_shift

            if not self.repeat:
                break


# ---------------------------------------------------------------------------
# Unseen accuracy callback
# ---------------------------------------------------------------------------

class UnseenAccuracyCallback(L.Callback):
    """
    At each val check, compute rule-following accuracy on unseen-key data using
    AUTOREGRESSIVE generation so the metric reflects actual generation quality
    rather than teacher-forced next-token prediction.

    For each batch: preprocess the first n_prompt_beats as a prompt, generate
    the next n_gen_beats autoregressively (greedy), then check whether the
    pitch class of each generated beat matches (key + OFFSETS[t%4]) % 12.

    The defaults are 1 + 23 so this matches the protocol the results table
    reports, and the integer experiment's. They used to be 4 + 16, which is
    a strictly easier task -- four beats is one whole I-IV-V-I cycle, so the
    prompt pins down key AND phase from content alone, while a one-beat
    prompt forces the model to get phase from position. Since ModelCheckpoint
    monitors this metric, the mismatch meant checkpoints were ranked on a
    protocol other than the one they were then scored under.
    """

    def __init__(self, dataloader: DataLoader, n_batches: int = 5,
                 n_prompt_beats: int = 1, n_gen_beats: int = 23,
                 beats_per_chord: int = 1):
        self.dataloader     = dataloader
        self.n_batches      = n_batches
        self.n_prompt_beats = n_prompt_beats
        self.n_gen_beats    = n_gen_beats
        self.beats_per_chord = beats_per_chord

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        # AR generation is expensive; only run on rank 0 to avoid DDP OOM / sync issues
        if trainer.global_rank != 0:
            return
        model       = pl_module.model
        device      = pl_module.device
        total_beats = self.n_prompt_beats + self.n_gen_beats

        correct    = 0
        total      = 0
        degenerate = 0          # beats that emitted no note at all

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= self.n_batches:
                    break
                x, pitch_shift = [t.to(device) for t in batch]

                prompt = model.base.preprocess(
                    x[:, :self.n_prompt_beats, :], pitch_shift
                )                                              # (B, n_prompt_beats, subseq)

                # Autoregressive generation — greedy (temperature=0)
                sampled = model.global_sampling(
                    prompt, max_seq_len=total_beats, temperature=0,
                )

                # Extract key from prompt's first beat: slot 1 % 128 % 12
                key = (prompt[:, 0, 1] % 128 % 12)            # (B,)

                # sampled[t] is (B, subseq_len) preprocessed token for beat t.
                # Slot 1 = pitch + (dur+1)*128  →  token % 128 = pitch  →  % 12 = pc
                # With one chord per bar only the downbeat is pinned to the
                # root; the other beats may legitimately be chord tones or rests.
                bpc = self.beats_per_chord
                for t in range(self.n_prompt_beats, total_beats):
                    if bpc > 1 and t % bpc != 0:
                        continue
                    y_t         = sampled[t]                   # (B, subseq)
                    # A beat that emitted no note must count as WRONG, not as
                    # whatever its control token happens to decode to. Taking
                    # slot 1 % 128 % 12 unconditionally scored pad(3330) as
                    # pc 2, eos(3329) as pc 1 and sos(3328) as pc 0, so one
                    # degenerate beat in twelve was counted CORRECT. Same
                    # validity test as evaluate_cp_bass._extract_pc, which the
                    # results table already used.
                    prog, pitch_dur = y_t[:, 0], y_t[:, 1]
                    valid       = (prog < 128) & (pitch_dur >= 128)
                    pred_pc     = (pitch_dur % 128) % 12      # (B,)
                    expected_pc = (key + OFFSETS[(t // bpc) % 4]) % 12
                    correct    += ((pred_pc == expected_pc) & valid).sum().item()
                    degenerate += (~valid).sum().item()
                    total      += len(pred_pc)

        acc = correct / max(total, 1)
        pl_module.log('unseen_acc', acc, prog_bar=True)
        # Surfaced because a rising degenerate rate means the adapter's write is
        # disturbing the local decoder's EOS decision rather than the pitch
        # field -- a different failure from simply naming the wrong root.
        pl_module.log('unseen_degenerate', degenerate / max(total, 1),
                      prog_bar=False)


# ---------------------------------------------------------------------------
# Lightning module wrapper
# ---------------------------------------------------------------------------

class CPYinyangLightning(L.LightningModule):

    def __init__(self, model: CPYinyangTransformer, max_lr: float, max_steps: int,
                 enc_loss_weight: float = 0.0):
        super().__init__()
        self.model           = model
        self.max_lr          = max_lr
        self.max_steps       = max_steps
        self.enc_loss_weight = enc_loss_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, pitch_shift = batch
        x_proc = self.model.base.preprocess(x, pitch_shift)
        loss   = self.model.loss(x, pitch_shift)

        if self.enc_loss_weight > 0:
            result = self.model.get_encoder_logits(x_proc)
            if result is not None:
                enc_logits, cur_pc = result            # (B,T,12), (B,T)
                enc_loss = F.cross_entropy(
                    enc_logits.reshape(-1, enc_logits.shape[-1]),
                    cur_pc.reshape(-1),
                )
                loss = loss + self.enc_loss_weight * enc_loss
                self.log('enc_loss', enc_loss, on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        # Log the LR but do NOT step: configure_optimizers hands the scheduler
        # to Lightning with interval='step', so Lightning owns the stepping.
        # Stepping here as well overran OneCycleLR -- see configure_optimizers.
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            self.log('training/lr', scheduler.get_last_lr()[0], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, pitch_shift = batch
        loss = self.model.loss(x, pitch_shift)
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
        # Returning the legacy ([optimizer], [scheduler]) pair makes Lightning
        # treat the scheduler as epoch-interval and step it at the end of every
        # training epoch -- on top of the manual step training_step used to do.
        # The finetune set is small (a few hundred batches), so a 10k-step run
        # ends ~19 epochs and overran OneCycleLR by ~19 calls, killing the run
        # about 19 steps short of max_steps. Padding total_steps only moved the
        # crash; the fix is to have exactly one stepper. Lightning is it, at
        # interval='step', which makes total_steps=max_steps exact.
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Unlike pretrain_cp_bass and finetune_cp_bass, this trainer was never
    # seeded, so every launch drew a different adapter init, data order and
    # dropout pattern. With one run per stem that left run-to-run variance
    # indistinguishable from a real effect -- frozen and lora swapped places
    # between two runs whose rule-model changes were provably bit-identical.
    L.seed_everything(args.seed, workers=True)
    n_gpus   = max(torch.cuda.device_count(), 1)
    max_lr   = 5e-5 if args.model_size >= 2 else 1e-4
    lora_suffix  = f'_lora{args.lora_rank}' if args.lora_rank > 0 else ''
    rule_suffix  = f'_{args.rule_mode}' if args.rule_mode != 'current' else ''
    enc_suffix   = f'_{args.encoder_type}' if args.encoder_injected else ''
    run_name = (
        args.run_name
        or f'cp_yinyang_size{args.model_size}_rank{args.adapter_rank}_skip{args.n_skip}{lora_suffix}{rule_suffix}{enc_suffix}'
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
        base_model        = base,
        adapter_rank      = args.adapter_rank,
        n_skip            = args.n_skip,
        lora_rank         = args.lora_rank,
        bidirectional     = args.bidirectional,
        encoder_injected  = args.encoder_injected,
        encoder_type      = args.encoder_type,
        rule_mode         = args.rule_mode,
        rule_view         = args.rule_view,
        rule_input        = args.rule_input,
        rule_enc_type     = args.rule_enc_type,
        rule_enc_hidden   = args.rule_enc_hidden,
        rule_key_select   = args.rule_key_select,
        rule_subbeats_per_chord = args.beats_per_chord,
        strict_causal     = args.strict_causal,
        gate_init         = args.gate_init,
        freeze_base_eval  = args.freeze_base_eval,
    )

    if args.unfreeze_base and args.freeze_base_eval:
        raise SystemExit('--freeze_base_eval is for a FROZEN host; with '
                         '--unfreeze_base the base is training and must stay '
                         'in train mode.')

    if args.unfreeze_base:
        for p in adapter.base.parameters():
            p.requires_grad_(True)
        print('Base model UNFROZEN — training base + adapter jointly')

    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in adapter.parameters() if not p.requires_grad)
    print(f'Trainable: {n_trainable:,}   Frozen: {n_frozen:,}')

    lit = CPYinyangLightning(adapter, max_lr=max_lr, max_steps=args.max_steps,
                             enc_loss_weight=args.enc_loss_weight)

    # Shared cache so datasets pointing to the same file reuse one tensor copy
    _cache: dict = {}

    # Window starts must land on a CYCLE boundary, because the rule model
    # derives phase from position: a window beginning mid-cycle would have
    # position 0 treated as phase 0 when it is not. A cycle is
    # SUBBEATS_PER_BAR * beats_per_chord beats -- 4 with one chord per beat,
    # 16 with one chord per bar -- so the default follows beats_per_chord
    # rather than being fixed at 4.
    if args.sample_step is None:
        args.sample_step = SUBBEATS_PER_BAR * args.beats_per_chord
    if args.sample_step % (SUBBEATS_PER_BAR * args.beats_per_chord):
        raise SystemExit(
            f'--sample_step {args.sample_step} is not a multiple of the '
            f'{SUBBEATS_PER_BAR * args.beats_per_chord}-beat cycle; windows '
            f'would start at a random phase and the rule model would be '
            f'misaligned for the whole window.')
    print(f'window alignment   : sample_step={args.sample_step} '
          f'(cycle = {SUBBEATS_PER_BAR * args.beats_per_chord} beats)')

    # See pretrain_cp_bass: sample_step aligns each window to a cycle
    # boundary so position encodes phase. Must match whatever the base was
    # pretrained with, or the adapter sees a different alignment than its host.
    train_ds = BassFramedDataset(args.train_data, args.train_length, args.batch_size,
                                 split=args.train_split,
                                 sample_step=args.sample_step)
    train_ds.preload(_cache)

    if args.pretrain_data and os.path.exists(args.pretrain_data):
        pretrain_ds = BassFramedDataset(args.pretrain_data, args.train_length, args.batch_size,
                                        split=args.train_split,
                                        sample_step=args.sample_step)
        pretrain_ds.preload(_cache)
        effective_train_ds = InterleavedDataset(pretrain_ds, train_ds)
        print(f'Training on interleaved: {args.pretrain_data}  +  {args.train_data}')
    else:
        effective_train_ds = train_ds

    val_ds = BassFramedDataset(args.val_data, args.train_length, args.batch_size,
                                split=args.val_split, sample_step=args.sample_step,
                                repeat=True)
    val_ds.preload(_cache)

    # num_workers=0: all data loading stays in the main process so _cache sharing works.
    # A worker subprocess would copy-on-write the tensor and defeat the sharing.
    train_loader = DataLoader(effective_train_ds, batch_size=None, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=0)

    val_loaders   = [val_loader]
    unseen_acc_cb = None
    if args.unseen_data and os.path.exists(args.unseen_data):
        unseen_ds = BassFramedDataset(args.unseen_data, args.train_length, args.batch_size,
                                       split='val', sample_step=args.sample_step,
                                       repeat=True)
        unseen_ds.preload(_cache)
        unseen_loader = DataLoader(unseen_ds, batch_size=None, num_workers=0)
        val_loaders.append(unseen_loader)
        unseen_acc_cb = UnseenAccuracyCallback(
            unseen_loader, n_batches=5,
            n_prompt_beats=args.probe_prompt_beats,
            n_gen_beats=args.probe_gen_beats,
            beats_per_chord=args.beats_per_chord)
        print(f'Unseen eval data: {args.unseen_data}')
        if args.probe_prompt_beats > args.train_length:
            raise SystemExit(
                f'--probe_prompt_beats {args.probe_prompt_beats} exceeds the '
                f'{args.train_length}-beat window, so the prompt slice is silently '
                f'truncated and unseen_acc scores the wrong beats.')
        print(f'unseen_acc probe   : {args.probe_prompt_beats}-beat prompt, '
              f'{args.probe_gen_beats} generated')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.unseen_data and os.path.exists(args.unseen_data):
        ckpt_monitor, ckpt_mode, ckpt_filename = 'unseen_acc', 'max', run_name + '.{epoch:02d}.{unseen_acc:.4f}'
    else:
        ckpt_monitor, ckpt_mode, ckpt_filename = 'val_loss',   'min', run_name + '.{epoch:02d}.{val_loss:.5f}'
    checkpoint_cb = L.callbacks.ModelCheckpoint(
        monitor           = ckpt_monitor,
        mode              = ckpt_mode,
        save_top_k        = 5,
        save_last         = True,
        save_weights_only = True,
        enable_version_counter = False,
        dirpath    = os.path.join(args.ckpt_dir, run_name),
        filename   = ckpt_filename,
    )

    loggers = []
    if args.wandb_project:
        loggers.append(WandbLogger(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            config  = vars(args),
        ))
    if not args.no_tensorboard:
        try:
            loggers.append(TensorBoardLogger('tb_logs', name=run_name))
        except ModuleNotFoundError as e:
            print(f'  TensorBoard logging off: {e.msg.splitlines()[0]}')

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
    p.add_argument('--base_ckpt',          type=str, default=None,
                   help='Path to pretrained CP transformer .pt file (omit to train from random weights)')
    p.add_argument('--train_data',         type=str, required=True,
                   help='Primary (finetune) training data .pt file')
    p.add_argument('--pretrain_data',      type=str, default=None,
                   help='Optional second data file; batches are interleaved 50/50 with --train_data')
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
    p.add_argument('--unfreeze_base',      action='store_true',
                   help='Unfreeze entire base model for joint base+adapter training (use with --pretrain_data when training from scratch)')
    p.add_argument('--bidirectional',     action='store_true',
                   help='No-input-to-rule-model variant: AR hidden states are projected to rule space via a learned linear instead of reading the key from the sequence')
    p.add_argument('--encoder_injected', action='store_true',
                   help='Replace the one-hot W_E pitch-class lookup with a learned encoder; W_pos stays frozen')
    p.add_argument('--encoder_type', type=str, default='embedding',
                   choices=['embedding', 'token_mlp'],
                   help='embedding: nn.Embedding(12, d_model); '
                        'token_mlp: one_hot(12)→Linear(64)→ReLU→Linear(d_model)')
    p.add_argument('--enc_loss_weight', type=float, default=0.0,
                   help='Auxiliary CE loss weight for encoder current-pitch-class reconstruction. '
                        'Directly supervises dims 0-11 of encoder output to predict the current pc. '
                        'Only active when --encoder_injected. Recommended: 1.0.')
    p.add_argument('--rule_mode', type=str, default='current',
                   choices=['current', 'period4', 'seed_broadcast', 'compiled'],
                   help='current: analytical 16-dim rule model (existing); '
                        'period4: TRACR attention 28-dim, dims 12-23=next pc; '
                        'seed_broadcast: 28-dim, dims 12-23=seed pc; '
                        'compiled: full TracR compilation (BassRaspCompiled) -- '
                        'attention head retrieves the key, MLP computes the next '
                        'root, so the rule is in the weights. Use --rule_view to '
                        'choose which subspaces the adapter reads.')
    p.add_argument('--rule_view', type=str, default='key_phase',
                   choices=['root_next', 'key_phase', 'computed', 'full'],
                   help='Only with --rule_mode compiled. Which subspaces become '
                        'rule_hidden: root_next (12) hands the adapter the answer, '
                        'so it only has to copy; key_phase (16) hands it the key '
                        'and the next phase, so it must learn '
                        '(key + OFFSETS[p]) %% 12; computed (32) is everything the '
                        'network computed; full (44) adds the raw token echo.')
    p.add_argument('--no_tensorboard', action='store_true',
                   help='Skip the TensorBoard logger. It is also skipped\n'
                        'automatically when neither tensorboard nor\n'
                        'tensorboardX is installed.')
    p.add_argument('--rule_input',         type=str, default='tokens',
                   choices=['tokens', 'ar_state'],
                   help="tokens: the rule model reads the pitch-class stream. "
                        "ar_state: NO external input -- the host's pre-block "
                        "state is encoded into the rule model's token subspace. "
                        "Requires --rule_mode compiled.")
    p.add_argument('--rule_enc_type',      type=str, default='mlp',
                   choices=['linear', 'mlp'],
                   help='Encoder for --rule_input ar_state. Output is softmaxed: '
                        'the compiled MLP gates pairs with ReLU(k + p - 1), which '
                        'needs every dimension in [0, 1].')
    p.add_argument('--rule_enc_hidden',    type=int, default=64,
                   help='Hidden width for --rule_enc_type mlp.')
    p.add_argument('--rule_key_select',    type=str, default='phase0',
                   choices=['phase0', 'first'],
                   help="How the compiled head retrieves the key. 'phase0' "
                        'matches every phase-0 slot and averages them -- a '
                        "clean one-hot on rule-following input, a blend off "
                        "it, and a violation of Tracr Appendix D.1's "
                        "one-token rule. 'first' matches only the slot "
                        'holding beat 0, so the head attends exactly one '
                        'token. Identical outputs on rule-following input; '
                        'they differ only when the stream breaks the rule.')
    p.add_argument('--seed',               type=int, default=42,
                   help='Seed for pl.seed_everything. Pass a distinct '
                        '--run_name per seed so the runs do not share a '
                        'checkpoint directory.')
    p.add_argument('--beats_per_chord',    type=int, default=1,
                   help='Beats the root is held for. 1 (default) is the '
                        'original data, where the root changes every beat and '
                        'a cycle fills one bar. 4 matches --cycle_unit bar: '
                        'the root is held for a whole bar, a cycle spans four '
                        'bars, and only the downbeat is pinned to it -- so the '
                        'rule model holds each phase for 4 steps and the probe '
                        'scores downbeats only.')
    p.add_argument('--train_length',       type=int, default=TRAIN_LENGTH,
                   help=f'Window length in beats (default {TRAIN_LENGTH}). With '
                        '--beats_per_chord 4 a cycle is 16 beats, so use 32 or '
                        'more to cover at least two cycles.')
    p.add_argument('--gate_init',          type=float, default=0.0,
                   help='Initial value of each adapter block\'s output gate. '
                        '0.0 (default) means the adapter contributes nothing '
                        'at step 0 and grows in -- the standard discipline for '
                        'an adapter residual branch, and the same convention '
                        'as lora_B=zeros. Runs before this flag existed used '
                        '1.0, which wrote a randomly-initialised cross-'
                        'attention into the host at full strength; pass 1.0 to '
                        'reproduce them.')
    p.add_argument('--freeze_base_eval',   action='store_true',
                   help='Hold the frozen host in eval mode during training, so '
                        'its dropout is off. Lightning otherwise puts the whole '
                        'module in train mode and the frozen base keeps '
                        'dropping at 0.1, which cannot regularise weights that '
                        'are not learning but does force the adapter to '
                        'tolerate noisy host states. Whether that helps is '
                        'open -- A/B it. Incompatible with --unfreeze_base.')
    p.add_argument('--strict_causal',      action='store_true',
                   help='Mask the adapter cross-attention to j < t instead of '
                        'j <= t, in training AND generation. By default the '
                        "query at t may read rule_hidden[t], the rule's "
                        'prediction FOR beat t -- which is derived from the '
                        'key and position, not from beat t, so it leaks '
                        'nothing. With this flag the answer for beat t is '
                        'withheld; under a period-4 rule the same root is '
                        'still reachable at j = t-4, so the adapter must '
                        'learn a -4 offset, and the first few generated beats '
                        'have no same-phase row to read at all.')
    p.add_argument('--sample_step',        type=int, default=None,
                   help='Align training window starts to a multiple of this. This '
                        'trainer always used SUBBEATS_PER_BAR (one I-IV-V-I cycle); '
                        'that is now the default rather than hardcoded. '
                        'pretrain_cp_bass defaults to 1, so the two can disagree -- '
                        'they must match or the adapter sees a different window '
                        'alignment than its host was trained on.')
    p.add_argument('--probe_prompt_beats', type=int, default=1,
                   help='Prompt length for the unseen_acc probe that drives '
                        'checkpoint selection. Keep it equal to what '
                        'cp_results_table reports with, or checkpoints are '
                        'ranked under a different protocol than they are '
                        'scored under. probe_prompt_beats + probe_gen_beats '
                        f'must not exceed the {TRAIN_LENGTH}-beat window.')
    p.add_argument('--probe_gen_beats',    type=int, default=23,
                   help='Beats generated by the unseen_acc probe.')
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
