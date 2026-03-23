# Modulo-Sequence RASP/Tracr Rule-Adapter

This project studies whether a frozen AR transformer can be made to follow a
modular arithmetic sequence rule by patching on a small cross-attention adapter
guided by a RASP-compiled rule model.

**The rule:** at position `i`, the token is `(x + OFFSETS[i % 4]) % 12`
where `x` is the starting integer and `OFFSETS = [0, 5, 7, 0]`.
Example for `x=2`: `2, 7, 9, 2, 2, 7, 9, 2, ...`

---

## Setup

```bash
pip install torch numpy

# Optional: RASP/tracr support (requires JAX)
pip install jax jaxlib
pip install git+https://github.com/google-deepmind/tracr.git
```

Without JAX/tracr the code automatically falls back to a pure-NumPy rule model
that implements the same rule.

---

## Training

### Option A: Full pipeline (recommended)

Runs pretrain → adapter training → evaluation in one command:

```bash
python main.py
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | `all` | Run one stage: `pretrain`, `adapter`, `eval`, or `all` |
| `--epochs_pretrain` | `200` | Epochs for AR pretraining |
| `--epochs_adapter` | `100` | Epochs for adapter training |
| `--n_cycles` | `8` | Sequence length = `4 * n_cycles` tokens |
| `--d_model` | `128` | Transformer hidden size |
| `--n_layers` | `4` | Number of transformer layers |
| `--n_heads` | `4` | Number of attention heads |
| `--ckpt_dir` | `checkpoints` | Directory to save checkpoints |
| `--force_fallback` | off | Use NumPy rule model (skip tracr/JAX) |

---

### Option B: Run each stage separately

**Step 1 — Pretrain the AR transformer** on starting integers `{0..7}`:

```bash
python training/pretrain.py \
    --epochs 200 \
    --d_model 128 --n_layers 4 --n_heads 4 \
    --n_cycles 8 \
    --ckpt_dir checkpoints
```

Saves: `checkpoints/ar_transformer.pt`

---

**Step 2 — Fine-tune baseline** (no rule alignment, for comparison):

```bash
python training/finetune.py \
    --ar_ckpt checkpoints/ar_transformer.pt \
    --epochs 200 \
    --lr 1e-4 \
    --ckpt_dir checkpoints
```

Saves: `checkpoints/ar_finetuned.pt`

---

**Step 3 — Train the adapter** (frozen AR + frozen rule model, only 17 parameters train):

```bash
python training/train_adapter.py \
    --ar_ckpt checkpoints/ar_transformer.pt \
    --epochs 100 \
    --lr 3e-3 \
    --kl_weight 1.0 \
    --ckpt_dir checkpoints
```

Saves: `checkpoints/adapter.pt`

Key adapter flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--kl_weight` | `1.0` | Weight on KL distillation loss toward rule model |
| `--adapter_rank` | `32` | Low-rank attention dimension (must be divisible by 4) |
| `--force_fallback` | off | Use NumPy rule model instead of tracr |

---

## Evaluation

Compares three models across four data splits and prints a results table:

```bash
python evaluate.py \
    --ar_ckpt      checkpoints/ar_transformer.pt \
    --ft_ckpt      checkpoints/ar_finetuned.pt \
    --adapter_ckpt checkpoints/adapter.pt \
    --verbose
```

Output example:

```
========================================================================
RULE-FOLLOWING ACCURACY  (generation from 1-token prompt)
========================================================================
Data Split           Starters             Pretrain   Finetune    Ft+Rule
------------------------------------------------------------------------
Pretrain-only        A\B = {0,1}             1.000      0.500      1.000
Finetune-only        B\A = {6,7}             0.000      1.000      1.000
Both                 A∩B = {2,3,4,5}         1.000      1.000      1.000
Neither              {8,9,10,11}             0.000      0.000      1.000
------------------------------------------------------------------------
Overall              all 12 starters         0.500      0.625      1.000
========================================================================
```

Evaluation flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt_len` | `1` | Number of prompt tokens fed before generation |
| `--verbose` | off | Print per-starter accuracy breakdown |
| `--n_cycles` | `8` | Generation length = `4 * n_cycles` tokens |

---

## Checkpoints

All checkpoints are saved under `checkpoints/`:

| File | Description |
|------|-------------|
| `ar_transformer.pt` | Pretrained AR model (starters 0–7) |
| `ar_finetuned.pt` | Fine-tuned AR model (no rule, starters 2–7) |
| `adapter.pt` | Trained adapter weights (17 parameters) |

---

## Project Structure

```
rasp/
├── main.py                      # End-to-end pipeline runner
├── evaluate.py                  # Evaluation and results table
├── training/
│   ├── pretrain.py              # Stage 1: AR transformer pretraining
│   ├── finetune.py              # Stage 2: Fine-tune baseline (no rule)
│   └── train_adapter.py         # Stage 3: Adapter training
├── models/
│   ├── transformer.py           # Autoregressive transformer
│   ├── rule_model.py            # Frozen rule model (PyTorch wrapper)
│   └── adapter.py               # Cross-attention adapter + PatchedModel
├── rasp_program/
│   └── sequence_rule.py         # RASP program + tracr compiler + fallback
├── data/
│   └── dataset.py               # Dataset and dataloaders
└── requirements.txt
```
