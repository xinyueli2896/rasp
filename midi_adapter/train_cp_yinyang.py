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
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer, FramedDataset
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS

DEFAULT_TRAIN_LENGTH = 64   # 4 bars × 16 subbeats at beat_div=4
MAX_STEPS            = 100_000


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
        # Paired-condition tensors — populated in preload() or __iter__().
        self.keys      = None
        self.chord_seq = None

    def _load_pair_files(self) -> None:
        """Load optional .chord_seq.pt / .keys.pt sidecars (chord_seq wins)."""
        cs_path   = self.file_path[:-3] + '.chord_seq.pt'
        keys_path = self.file_path[:-3] + '.keys.pt'
        if os.path.exists(cs_path):
            self.chord_seq = torch.load(cs_path, weights_only=True).long()
        elif os.path.exists(keys_path):
            self.keys = torch.load(keys_path, weights_only=True).long()

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

        self._load_pair_files()
        if self.chord_seq is not None:
            print(f'  paired chord_seq: {tuple(self.chord_seq.shape)}')
        elif self.keys is not None:
            print(f'  paired keys: {len(self.keys)}')

    def __len__(self):
        # A repeating iterator produces samples indefinitely. Reporting the
        # single-pass batch count made DataLoader spam a warning each step
        # once we passed that count. Return a huge number so DataLoader
        # thinks we always have more to yield; Lightning uses
        # val_check_interval, not __len__, to schedule val loops.
        if self.repeat:
            return 2**31 - 1
        return super().__len__()

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
            self._load_pair_files()
            msg = f'Data for dataset {self.file_path} loaded.'
            if self.chord_seq is not None:
                msg += f' Paired chord_seq: {tuple(self.chord_seq.shape)}'
            elif self.keys is not None:
                msg += f' Paired keys: {len(self.keys)}'
            print(msg)

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

                if self.chord_seq is not None:
                    yield (self.data[index_matrix], pitch_shift,
                           self.chord_seq[raw_ids])
                elif self.keys is not None:
                    yield self.data[index_matrix], pitch_shift, self.keys[raw_ids]
                else:
                    yield self.data[index_matrix], pitch_shift

            if not self.repeat:
                break


# ---------------------------------------------------------------------------
# Unseen accuracy callback
# ---------------------------------------------------------------------------

class RuleFollowingCallback(L.Callback):
    """
    At each val check, run AUTOREGRESSIVE greedy generation on a val loader and
    measure how often the generated content obeys the I-IV-V-I chord rule.

    Two metrics are logged per prefix:

      {prefix}_bass_acc   fraction of generated subbeats where voice 0's pitch
                          class == expected chord root (bass-line rule)
      {prefix}_chord_acc  fraction of generated subbeats where the expected
                          major triad {root, root+4, root+7} ⊆ pitch classes
                          across ALL voices (harmony rule)

    `{prefix}_acc` is aliased to `{prefix}_bass_acc` for backwards compat.

    The expected root at each generated subbeat is computed from the batch's
    paired signal — the ground-truth key we trained on — NOT from the prompt's
    first pitch class (which may not be the tonic). This makes the metric the
    honest "does the model follow the rule we told it to follow" test.

    beats_per_bar = subbeats per chord slot at beat_div=4:
        16 → 1 chord per bar,  8 → 2 chords per bar (default at chords_per_bar=2).
    """

    def __init__(self, dataloader: DataLoader, prefix: str = 'unseen',
                 n_batches: int = 5, n_prompt_beats: int = 4,
                 n_gen_beats: int = 16, beats_per_bar: int = 1):
        self.dataloader     = dataloader
        self.prefix         = prefix
        self.n_batches      = n_batches
        self.n_prompt_beats = n_prompt_beats
        self.n_gen_beats    = n_gen_beats
        self.beats_per_bar  = beats_per_bar

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.global_rank != 0:   # AR gen is expensive; rank 0 only
            return
        model       = pl_module.model
        device      = pl_module.device
        total_beats = self.n_prompt_beats + self.n_gen_beats
        tokenizer   = model.base.tokenizer

        bass_correct  = 0
        chord_correct = 0
        total         = 0

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= self.n_batches:
                    break
                tensors = [t.to(device) for t in batch]
                x, pitch_shift = tensors[0], tensors[1]

                # Route the 3rd tensor by shape; also derive the ground-truth
                # key per sample from whichever paired signal is present.
                chord_seq    = None
                key_from_pair = None
                if len(tensors) >= 3:
                    if tensors[2].dim() == 2:
                        chord_seq     = (tensors[2] + pitch_shift.unsqueeze(-1)) % 12
                        key_from_pair = chord_seq[:, 0].long()   # first chord IS the tonic (phase=0)
                    else:
                        key_from_pair = (tensors[2] + pitch_shift) % 12

                # Bidirectional derives rule_hidden from its own AR states;
                # global_sampling would ignore chord_seq anyway, but drop it
                # here so the no-rule-input claim is explicit at the call site.
                if model.bidirectional:
                    chord_seq = None

                prompt = model.base.preprocess(
                    x[:, :self.n_prompt_beats, :], pitch_shift
                )   # (B, n_prompt_beats, subseq)

                sampled = model.global_sampling(
                    prompt, max_seq_len=total_beats, temperature=0,
                    chord_seq=chord_seq,
                )

                # Fallback: derive from prompt's first pitch class.
                if key_from_pair is None:
                    key_from_pair = (prompt[:, 0, 1] % 128) % 12   # (B,)

                # Iterate over generated subbeats and score both metrics.
                for t in range(self.n_prompt_beats, total_beats):
                    y_t   = sampled[t]                    # (B, subseq)
                    B, S  = y_t.shape

                    phase        = (t // self.beats_per_bar) % 4
                    expected_root = (key_from_pair + OFFSETS[phase]) % 12   # (B,)

                    # Bass rule — voice 0's pitch class
                    pred_bass_pc = (y_t[:, 1] % 128) % 12
                    bass_correct += (pred_bass_pc == expected_root).sum().item()

                    # Chord rule — expected triad ⊆ pitch classes across all voices
                    for b in range(B):
                        pcs: set[int] = set()
                        for v in range(0, S, 2):
                            a = int(y_t[b, v].item())
                            if a == tokenizer.eos_token or a == tokenizer.pad_token:
                                break
                            if v + 1 >= S:
                                break
                            bt = int(y_t[b, v + 1].item())
                            if bt == tokenizer.eos_token or bt == tokenizer.pad_token or bt < 128:
                                continue
                            pcs.add((bt % 128) % 12)
                        r = int(expected_root[b].item())
                        expected_triad = {(r + i) % 12 for i in (0, 4, 7)}
                        if expected_triad.issubset(pcs):
                            chord_correct += 1

                    total += B

        bass_acc  = bass_correct  / max(total, 1)
        chord_acc = chord_correct / max(total, 1)
        pl_module.log(f'{self.prefix}_bass_acc',  bass_acc,  prog_bar=True)
        pl_module.log(f'{self.prefix}_chord_acc', chord_acc, prog_bar=False)
        # Legacy alias so existing checkpoints / dashboards keep working
        pl_module.log(f'{self.prefix}_acc',       bass_acc,  prog_bar=False)


# Back-compat alias
UnseenAccuracyCallback = RuleFollowingCallback


# ---------------------------------------------------------------------------
# Lightning module wrapper
# ---------------------------------------------------------------------------

class BaseFinetuneWrapper(nn.Module):
    """Full fine-tuning baseline: the plain pretrained CP transformer with NO
    adapter and NO rule conditioning. The model can only infer the key from
    the prompt. Exposes the same .loss / .global_sampling / .base interface as
    CPYinyangTransformer so the Lightning module and rule callbacks work
    unchanged (paired condition tensors are accepted and ignored)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.approach = 'chord'   # keeps downstream attribute reads happy

    def loss(self, x, pitch_shift, key_override=None, chord_seq=None):
        return self.base.loss(x, pitch_shift)

    def global_sampling(self, x, max_seq_len=384, temperature=1.0,
                         sampling_func=None, chord_seq=None):
        return self.base.global_sampling(x, max_seq_len=max_seq_len,
                                          temperature=temperature,
                                          sampling_func=sampling_func)


class CPYinyangLightning(L.LightningModule):

    def __init__(self, model: CPYinyangTransformer, max_lr: float, max_steps: int,
                 enc_loss_weight: float = 0.0, proxy_loss_weight: float = 0.0,
                 weight_decay: float = 1e-4, equiv_loss_weight: float = 0.0,
                 unseen_keys: tuple = (6, 8), equiv_shifts: tuple = (-1, 1)):
        super().__init__()
        self.model             = model
        self.max_lr            = max_lr
        self.max_steps         = max_steps
        self.enc_loss_weight   = enc_loss_weight
        self.proxy_loss_weight = proxy_loss_weight
        self.weight_decay      = weight_decay
        self.equiv_loss_weight = equiv_loss_weight
        self.unseen_keys       = tuple(unseen_keys)
        self.equiv_shifts      = tuple(equiv_shifts)

    def forward(self, x):
        return self.model(x)

    def _unpack(self, batch):
        """→ (x, pitch_shift, key_override, chord_seq).

        Datasets ship both a .keys.pt and a .chord_seq.pt sidecar and the loader
        prefers chord_seq, so bidirectional runs — which must NOT be handed a
        chord sequence — recover the tonic from it instead: chord slot 0 is
        phase 0, i.e. the I chord, so its root IS the key.
        """
        key_override, chord_seq = None, None
        if len(batch) == 3:
            x, pitch_shift, extra = batch
            if extra.dim() == 2:
                chord_seq = extra          # (B, N_chords)
            else:
                key_override = extra       # (B,)
        else:
            x, pitch_shift = batch
        if self.model.bidirectional and chord_seq is not None:
            key_override, chord_seq = chord_seq[:, 0].long(), None
        return x, pitch_shift, key_override, chord_seq

    def training_step(self, batch, batch_idx):
        x, pitch_shift, key_override, chord_seq = self._unpack(batch)
        x_proc = self.model.base.preprocess(x, pitch_shift)
        loss   = self.model.loss(x, pitch_shift,
                                  key_override=key_override,
                                  chord_seq=chord_seq)

        if self.enc_loss_weight > 0:
            result = self.model.get_encoder_logits(x_proc)
            if result is not None:
                enc_logits, target = result
                if self.model.approach == 'chord':
                    # target: (B,T,12) binary chromagram — use BCE loss
                    enc_loss = F.binary_cross_entropy_with_logits(enc_logits, target)
                else:
                    # target: (B,T) long current pitch class — use CE loss
                    enc_loss = F.cross_entropy(
                        enc_logits.reshape(-1, enc_logits.shape[-1]),
                        target.reshape(-1),
                    )
                loss = loss + self.enc_loss_weight * enc_loss
                self.log('enc_loss', enc_loss, on_step=True, on_epoch=False)

        # Bidirectional ("no rule input") mode: pull each layer's ar_to_rule
        # proxy toward the analytical rule hidden. Train-time only.
        if self.proxy_loss_weight > 0 and self.model._proxy_loss is not None:
            proxy_loss = self.model._proxy_loss
            loss = loss + self.proxy_loss_weight * proxy_loss
            self.log('proxy_loss', proxy_loss, on_step=True, on_epoch=False)

        # Transposition equivariance on the ar_to_rule chroma read. Shifts are
        # drawn only between SEEN keys, so no held-out key is ever fed in, and
        # the shifted copy carries no label of its own.
        if self.equiv_loss_weight > 0 and key_override is not None:
            equiv = self.model.equivariance_loss(
                x, pitch_shift, key_override,
                unseen_keys=self.unseen_keys, shifts=self.equiv_shifts)
            if equiv is not None:
                loss = loss + self.equiv_loss_weight * equiv
                self.log('equiv_loss', equiv, on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
            self.log('training/lr', scheduler.get_last_lr()[0], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, pitch_shift, key_override, chord_seq = self._unpack(batch)
        loss = self.model.loss(x, pitch_shift,
                                key_override=key_override,
                                chord_seq=chord_seq)
        key  = 'val_loss' if dataloader_idx == 0 else 'unseen_loss'
        self.log(key, loss, on_step=False, on_epoch=True,
                 sync_dist=True, add_dataloader_idx=False)
        return loss

    def configure_optimizers(self):
        # Adapter-finetune defaults (differ from upstream pretraining):
        # - weight_decay=1e-4: light regularization on the small trainable set
        # - pct_start=0.02: longer warmup helps when the only trainable modules
        #   are randomly initialized adapters bolted onto a fixed base
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.max_lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr,
            total_steps=self.max_steps, pct_start=0.02,
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.equiv_loss_weight > 0:
        # The constraint acts on ar_to_rule, which only exists in the
        # no-rule-input variant, and it reads ONE depth so the shifted pass is
        # cheap and unambiguous.
        if not args.bidirectional:
            raise SystemExit('--equiv_loss_weight requires --bidirectional: '
                             'there is no ar_to_rule read to constrain without it')
        if args.rule_from_layer < 0:
            raise SystemExit('--equiv_loss_weight requires --rule_from_layer >= 0 '
                             '(0 reads the stream entering the stack, where the '
                             'shifted pass costs only the local encoder)')
    n_gpus   = max(torch.cuda.device_count(), 1)
    max_lr   = 5e-5 if args.model_size >= 2 else 1e-4
    lora_suffix     = f'_lora{args.lora_rank}' if args.lora_rank > 0 else ''
    rule_suffix     = f'_{args.rule_mode}' if args.rule_mode != 'current' else ''
    enc_suffix      = f'_{args.encoder_type}' if args.encoder_injected else ''
    approach_suffix = f'_{args.approach}' if args.approach != 'bass' else ''
    bidir_suffix    = ('_bidir'
                       + ('_tracr' if args.rule_attention else '')
                       + (f'_h{args.rule_heads}' if args.rule_heads > 1 else '')
                       + (f'_L{args.rule_from_layer}'
                          if args.rule_from_layer >= 0 else '')
                       + (f'_mlp{args.ar_to_rule_hidden}'
                          if args.ar_to_rule_hidden > 0 else '')
                       + ('_full' if args.rule_program == 'full' else '')
                       + ('_triad' if args.rule_input == 'triad' else '')
                       + ('' if args.content_residual == 'none'
                          else f'_cr{args.content_residual}')
                       + ('' if args.proxy_activation == 'none'
                          else f'_{args.proxy_activation}')
                       + (f'_proxy{args.proxy_loss_weight:g}'
                          if args.proxy_loss_weight > 0 else '')
                       + (f'_equiv{args.equiv_loss_weight:g}'
                          if args.equiv_loss_weight > 0 else '')
                       ) if args.bidirectional else ''
    run_name = (
        args.run_name
        or f'cp_yinyang_size{args.model_size}_rank{args.adapter_rank}_skip{args.n_skip}'
           f'{lora_suffix}{rule_suffix}{enc_suffix}{approach_suffix}{bidir_suffix}'
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

    if args.finetune_base:
        # Full fine-tuning baseline — no adapter, no rule conditioning.
        adapter = BaseFinetuneWrapper(base)
        for p in adapter.parameters():
            p.requires_grad_(True)
        max_lr = args.finetune_lr
        print(f'FULL FINE-TUNE mode: training the entire base model, lr={max_lr}')
    else:
        adapter = CPYinyangTransformer(
            base_model        = base,
            adapter_rank      = args.adapter_rank,
            n_skip            = args.n_skip,
            lora_rank         = args.lora_rank,
            bidirectional     = args.bidirectional,
            encoder_injected  = args.encoder_injected,
            encoder_type      = args.encoder_type,
            rule_mode         = args.rule_mode,
            approach          = args.approach,
            chords_per_bar    = args.chords_per_bar,
            chord_seq_conditioning = args.paired_chord_seq,
            positional_qk     = args.positional_qk,
            qk_content_residual = args.qk_content_residual,
            content_residual  = args.content_residual,
            proxy_supervision = args.proxy_loss_weight > 0,
            rule_attention    = args.rule_attention,
            proxy_pos_inject  = not args.no_proxy_pos_inject,
            rule_heads        = args.rule_heads,
            rule_from_layer   = args.rule_from_layer,
            rule_program      = args.rule_program,
            rule_input        = args.rule_input,
            ar_to_rule_hidden = args.ar_to_rule_hidden,
            equiv_loss_kind   = args.equiv_loss_kind,
            proxy_activation  = args.proxy_activation,
            proxy_temp        = args.proxy_temp,
        )

        if args.unfreeze_base:
            for p in adapter.base.parameters():
                p.requires_grad_(True)
            print('Base model UNFROZEN — training base + adapter jointly')

    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in adapter.parameters() if not p.requires_grad)
    print(f'Trainable: {n_trainable:,}   Frozen: {n_frozen:,}')

    lit = CPYinyangLightning(adapter, max_lr=max_lr, max_steps=args.max_steps,
                             enc_loss_weight=args.enc_loss_weight,
                             proxy_loss_weight=args.proxy_loss_weight,
                             weight_decay=args.weight_decay,
                             equiv_loss_weight=args.equiv_loss_weight,
                             unseen_keys=tuple(args.unseen_keys),
                             equiv_shifts=tuple(args.equiv_shifts))

    # Shared cache so datasets pointing to the same file reuse one tensor copy
    _cache: dict = {}

    train_ds = BassFramedDataset(args.train_data, args.train_length, args.batch_size,
                                  split=args.train_split, sample_step=SUBBEATS_PER_BAR)
    train_ds.preload(_cache)

    if args.pretrain_data and os.path.exists(args.pretrain_data):
        pretrain_ds = BassFramedDataset(args.pretrain_data, args.train_length, args.batch_size,
                                        split=args.train_split, sample_step=SUBBEATS_PER_BAR)
        pretrain_ds.preload(_cache)
        effective_train_ds = InterleavedDataset(pretrain_ds, train_ds)
        print(f'Training on interleaved: {args.pretrain_data}  +  {args.train_data}')
    else:
        effective_train_ds = train_ds

    val_ds = BassFramedDataset(args.val_data, args.train_length, args.batch_size,
                                split=args.val_split, sample_step=SUBBEATS_PER_BAR,
                                repeat=True)
    val_ds.preload(_cache)

    # num_workers=0: all data loading stays in the main process so _cache sharing works.
    # A worker subprocess would copy-on-write the tensor and defeat the sharing.
    train_loader = DataLoader(effective_train_ds, batch_size=None, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=0)

    val_loaders    = [val_loader]
    # Beats-per-bar (= subbeats per chord slot) for the rule-following metric.
    if args.approach == 'chord':
        subs_per_chord = 16 // args.chords_per_bar   # 8 at chords_per_bar=2
    else:
        subs_per_chord = 4                            # bass approach: one root per beat

    # Rule-following callback on the SEEN val set → seen_bass_acc / seen_chord_acc.
    seen_acc_cb = RuleFollowingCallback(
        val_loader, prefix='seen',
        n_batches=5,
        n_prompt_beats=args.train_length // 4,
        n_gen_beats   =args.train_length - args.train_length // 4,
        beats_per_bar =subs_per_chord,
    )

    unseen_acc_cb = None
    if args.unseen_data and os.path.exists(args.unseen_data):
        unseen_ds = BassFramedDataset(args.unseen_data, args.train_length, args.batch_size,
                                       split='val', sample_step=SUBBEATS_PER_BAR,
                                       repeat=True)
        unseen_ds.preload(_cache)
        unseen_loader = DataLoader(unseen_ds, batch_size=None, num_workers=0)
        val_loaders.append(unseen_loader)
        unseen_acc_cb = RuleFollowingCallback(
            unseen_loader, prefix='unseen',
            n_batches=5,
            n_prompt_beats=args.train_length // 4,
            n_gen_beats   =args.train_length - args.train_length // 4,
            beats_per_bar =subs_per_chord,
        )
        print(f'Unseen eval data: {args.unseen_data}')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    # Two parallel checkpoint series — lowest val_loss (seen) and highest
    # unseen_bass_acc (rule-following on unseen keys).
    checkpoint_loss_cb = L.callbacks.ModelCheckpoint(
        monitor           = 'val_loss',
        mode              = 'min',
        save_top_k        = 5,
        save_last         = True,
        save_weights_only = True,
        enable_version_counter = False,
        dirpath    = os.path.join(args.ckpt_dir, run_name),
        filename   = run_name + '.by_val_loss.{epoch:02d}.{val_loss:.5f}',
    )
    checkpoint_cbs = [checkpoint_loss_cb]
    if unseen_acc_cb is not None:
        checkpoint_acc_cb = L.callbacks.ModelCheckpoint(
            monitor           = 'unseen_bass_acc',
            mode              = 'max',
            save_top_k        = 5,
            save_last         = False,   # save_last already produced by the val_loss series
            save_weights_only = True,
            enable_version_counter = False,
            dirpath    = os.path.join(args.ckpt_dir, run_name),
            filename   = run_name + '.by_unseen_acc.{epoch:02d}.{unseen_bass_acc:.4f}',
        )
        checkpoint_cbs.append(checkpoint_acc_cb)

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

    callbacks = list(checkpoint_cbs)
    callbacks.append(seen_acc_cb)
    if unseen_acc_cb is not None:
        callbacks.append(unseen_acc_cb)

    # Always clip at 1.0 for adapter finetuning — random-init adapters bolted
    # onto a frozen base can spike, so the clip guards stability even at size 1.
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

    # Load best (by val_loss) checkpoint and save adapter weights as plain .pt
    best_ckpt = checkpoint_loss_cb.best_model_path or checkpoint_loss_cb.last_model_path
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
    p.add_argument('--train_length',       type=int, default=DEFAULT_TRAIN_LENGTH,
                   help='Subbeats per training window. Must be ≤ the shortest song '
                        'in the dataset. Default 64 (= 4 bars at beat_div=4).')
    p.add_argument('--val_check_interval', type=int, default=500)
    p.add_argument('--adapter_rank',       type=int, default=256)
    p.add_argument('--n_skip',             type=int, default=4)
    p.add_argument('--lora_rank',          type=int, default=0,
                   help='LoRA rank for base model Q/V projections (0 = frozen base)')
    p.add_argument('--finetune_base',      action='store_true',
                   help='Full fine-tuning baseline: train the ENTIRE pretrained base '
                        'with no adapter and no rule conditioning. The model can only '
                        'infer the key from the prompt. Evaluate with '
                        'evaluate_on_real --no_adapter --base_ckpt <this run\'s ckpt>.')
    p.add_argument('--finetune_lr',        type=float, default=1e-5,
                   help='Learning rate for --finetune_base (default 1e-5, lower than '
                        'adapter training since all 100M+ base weights move).')
    p.add_argument('--unfreeze_base',      action='store_true',
                   help='Unfreeze entire base model for joint base+adapter training (use with --pretrain_data when training from scratch)')
    p.add_argument('--bidirectional',     action='store_true',
                   help='No-input-to-rule-model variant: AR hidden states are projected to rule space via a learned linear instead of reading the key from the sequence')
    p.add_argument('--rule_attention', action='store_true',
                   help='Use the compiled TracR-style chord rule model '
                        '(ChordTracrRuleModel, d_model=28): one FROZEN attention '
                        'head retrieves the tonic from phase-0 slots, leaving '
                        'root = (key + OFFSETS[phase]) %% 12 for the adapter to '
                        'compute. Requires --bidirectional. The frozen head '
                        'constrains the learned ar_to_rule proxy by construction, '
                        'so --proxy_loss_weight can be left at 0 — this is the '
                        'structural match to the integer experiment.')
    p.add_argument('--no_proxy_pos_inject', action='store_true',
                   help='With --rule_attention, do NOT add the frozen phase '
                        'encoding to the proxy — make ar_to_rule recover bar '
                        'phase from the base hidden states on its own.')
    p.add_argument('--rule_input', type=str, default='root',
                   choices=['root', 'triad'],
                   help="What dims 0-11 of the rule stream mean. 'root' is a "
                        "one-hot pitch class. 'triad' is a 12-d major-triad "
                        "chromagram -- the MLP transposes each active dim by "
                        "OFFSETS[phase], so a 3-hot tonic triad comes out as "
                        "the triad of the correct root, and the adapter is "
                        "handed a chord rather than a root. Requires "
                        "--rule_program full. Exact only for a CLEAN 3-hot "
                        "input: a 4-hot chromagram fires a 4th MLP cell and "
                        "adds a wrong note, so pair it with "
                        "--proxy_activation hard.")
    p.add_argument('--rule_program', type=str, default='retrieve',
                   choices=['retrieve', 'full', 'mlp_only'],
                   help="How much of the rule the compiled model executes. "
                        "'retrieve' (default) compiles only Aggregate -- one "
                        "attention head that fetches the tonic -- and leaves "
                        "root = (key + OFFSETS[phase]) %% 12 to the adapter. "
                        "'full' is the conventional TracR compilation: the "
                        "same head plus the MLP that SequenceMap compiles to, "
                        "so the program emits the correct root and the adapter "
                        "only has to perceive the key and render notes. "
                        "'mlp_only' drops the attention head: the projection "
                        "estimates the KEY directly from the full causal "
                        "context and the MLP applies the rule. Aggregate is "
                        "essential when the key must be found among supplied "
                        "chords, but with no input it forces the estimate to "
                        "come from phase-0 positions -- the earliest and least "
                        "informed, the first having seen no music at all -- "
                        "while discarding the context-rich later ones. "
                        "d_model 28 -> 40. Requires --rule_attention.")
    p.add_argument('--content_residual', type=str, default='none',
                   choices=['none', 'q', 'k', 'qk'],
                   help="Which side of the positional Q/K gets a zero-init "
                        "learned content term. 'q' lets the MUSIC shape what it "
                        "asks for (anticipation, adaptive strength) while the "
                        "keys stay pure addresses — the recommended setting. "
                        "'k' adds rule content to the keys, which makes two "
                        "slots carrying the same chord look alike and can pull "
                        "a query to the wrong one. 'qk' is the legacy "
                        "--qk_content_residual behaviour. Requires "
                        "--positional_qk; must match at eval.")
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='AdamW weight decay on the trainable set. 1e-4 is light; '
                        'raise it when the adapter overfits a small dataset.')
    p.add_argument('--ar_to_rule_hidden', type=int, default=0,
                   help='Hidden width of the AR->rule projection. 0 (default) '
                        'is a plain Linear(768 -> 12). A positive value makes '
                        'it Linear-ReLU-Linear, which matters most with '
                        '--rule_from_layer 0: the pre-stack hidden state has '
                        'had no self-attention, so naming a chord from it is a '
                        'nonlinear job. Must match at eval.')
    p.add_argument('--rule_from_layer', type=int, default=-1,
                   help='Where the no-input rule signal is read from. -1 '
                        '(default) = one projection per adapter, each reading '
                        'its own layer. 0 = ONE projection reading the stream '
                        'entering the stack, shared by all adapters — the '
                        'structural match to the explicit-input variant, where '
                        'rule_hidden is also built once. k = once after layer k '
                        '(must be <= n_skip so it exists before the first '
                        'adapter). Requires --bidirectional.')
    p.add_argument('--rule_heads', type=int, default=1, choices=[1, 2, 3, 4],
                   help='Phase heads in the compiled rule model. 1 reads the '
                        'key only off the I chords (one source in a 1-bar '
                        'prompt). 4 adds a head per phase that un-rotates its '
                        'own slots by -OFFSETS[p], recovering the same key '
                        'from every slot type — same answer on a clean proxy, '
                        'robust to a misread slot. Requires --rule_attention.')
    p.add_argument('--proxy_activation', type=str, default='none',
                   choices=['none', 'softmax', 'sigmoid', 'hard'],
                   help="Shape ar_to_rule's 12-d root output before the frozen "
                        "head reads it. 'softmax' normalises it to a "
                        "distribution so its scale matches W_E[root] in the "
                        "explicit-input variant; 'sigmoid' is per-dimension and "
                        "is the right choice when dims 0-11 are a CHROMAGRAM, "
                        "since softmax would force a four-note chord to ~0.25 "
                        "per note; 'hard' is straight-through one-hot. Any "
                        "program with an MLP REQUIRES one of these: the hidden "
                        "layer is ReLU(key_i + phase_j - 1), exact for key in "
                        "[0,1] and corrupted by raw logits. Must match at eval "
                        "— it changes the forward pass, not the weights.")
    p.add_argument('--proxy_temp', type=float, default=1.0,
                   help='Softmax temperature for --proxy_activation. Below 1 '
                        'sharpens toward one-hot; only read when the activation '
                        'is softmax or hard.')
    p.add_argument('--proxy_loss_weight', type=float, default=0.0,
                   help='Weight of the auxiliary loss pulling each layer\'s '
                        'ar_to_rule proxy toward the analytical rule hidden '
                        '(BCE on the triad chromagram + CE on bar phase). '
                        'Requires --bidirectional. 0 = leave ar_to_rule '
                        'unconstrained, which lets the adapter collapse into '
                        'plain self-attention. Recommended: 1.0. Supervision '
                        'is train-time only — inference needs no rule input.')
    p.add_argument('--equiv_loss_weight', type=float, default=0.0,
                   help='Weight of the transposition-equivariance constraint '
                        'on the ar_to_rule chroma read: transposing the input '
                        'by s semitones must rotate the 12 root logits by s. '
                        'Requires --bidirectional and --rule_from_layer >= 0. '
                        'This is what --proxy_loss_weight alone cannot give '
                        'you: CE on ten keys is satisfied by memorising ten '
                        'key-to-root tables, which says nothing about F#/G#, '
                        'whereas equivariance is a property of the MAP and so '
                        'transfers to every key. Shifts are drawn only between '
                        'SEEN keys (see --unseen_keys), so no held-out key is '
                        'ever fed in; the group is generated by s=+1, which '
                        'survives that filter. Train-time only. Recommended: '
                        '1.0 alongside --proxy_loss_weight 1.0.')
    p.add_argument('--equiv_loss_kind', type=str, default='js',
                   choices=['js', 'mse'],
                   help='How --equiv_loss_weight measures the mismatch. js = '
                        'Jensen-Shannon between the two root distributions, in '
                        'nats, so it is on the same footing as the proxy CE and '
                        'weight 1.0 binds; invariant to a global rescaling of '
                        'the logits. mse = squared error on the raw logits, '
                        'which is what the first run used and which did NOT '
                        'bind: at ~0.05 against an LM loss of ~1.5 it was 1.5% '
                        'of the objective, fell to 0.035 and then drifted back '
                        'to 0.055. Kept only for that comparison.')
    p.add_argument('--unseen_keys', type=int, nargs='*', default=[6, 8],
                   help='Tonics held out of training (default F#=6, G#=8). '
                        '--equiv_loss_weight never transposes INTO these, which '
                        'is what keeps the constraint leak-free.')
    p.add_argument('--equiv_shifts', type=int, nargs='+', default=[-1, 1],
                   help='Candidate semitone shifts for --equiv_loss_weight. '
                        '+-1 is enough: the cyclic group is generated by 1, so '
                        'equivariance under it implies equivariance under all '
                        '12. Larger shifts are more likely to be rejected for '
                        'pushing a note out of MIDI range.')
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
                   choices=['current', 'period4', 'seed_broadcast'],
                   help='current: analytical 16-dim rule model (existing); '
                        'period4: TRACR attention 28-dim, dims 12-23=next pc; '
                        'seed_broadcast: 28-dim, dims 12-23=seed pc.')
    p.add_argument('--approach', type=str, default='bass',
                   choices=['bass', 'chord'],
                   help='bass: regulate the bass note (slot 0 pitch class) with BassTracrRuleModel; '
                        'chord: regulate the chord progression (all-voice chromagram) with '
                        'CPChordRuleModel. Use --encoder_injected to inject a learned '
                        'ChordEncoder for chord approach training.')
    p.add_argument('--chords_per_bar', type=int, default=2, choices=[1, 2, 4],
                   help='Harmonic rhythm. 2 (default) = chord changes every half-bar (8 '
                        'subbeats at beat_div=4); 1 = chord per bar. Must match the value '
                        'used to filter the training data.')
    p.add_argument('--positional_qk', action='store_true',
                   help='Adapter cross-attention uses PURELY positional Q/K scaled '
                        'x20 (the integer-experiment recipe): temporal alignment is '
                        'hardcoded on the diagonal (strided in chord-seq mode) '
                        'instead of learned from content. Content flows only '
                        'through V.')
    p.add_argument('--qk_content_residual', action='store_true',
                   help='Only with --positional_qk. Adds ZERO-INITIALISED content '
                        'projections on top of the scaled positional Q/K, so routing '
                        'starts exactly positional but the music content can learn to '
                        'modulate it (anticipation, adaptive strength).')
    p.add_argument('--paired_chord_seq', action='store_true',
                   help='Use the explicit chord-root sequence (from .chord_seq.pt) as the '
                        'rule input, via ChordSeqRuleModel. Cross-attention runs at '
                        'chord-position granularity on the rule side (T_k = n_bars * '
                        'chords_per_bar) and the model learns the alignment to subbeat-'
                        'level queries. Mutually exclusive with --encoder_injected.')
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
