# Modulo-Sequence Yin-Yang Rule Adapter

This project studies whether a frozen AR transformer can be steered to follow a
modular arithmetic sequence rule by patching on cross-attention adapters guided
by a Tracr-equivalent rule model (the Yin-Yang architecture).

**The rule:** at position `i`, the token is `(x + OFFSETS[i % 4]) % 24`
where `x` is the starting integer and `OFFSETS = [0, 5, 7, 0]`.
Example for `x=2`: `2, 7, 9, 2, 2, 7, 9, 2, ...`

---

## Architecture

```
Yin  = rule model (frozen, no parameters)     → rule_hidden (B, T, 28)
Yang = AR transformer (frozen or LoRA)        → hidden states at each layer

Every n_skip AR layers:
    x = ar_block(x)
    x = x + yinyang_attn(query=x, key=rule_hidden, value=rule_hidden)
```

### Rule model

`models/tracr_pytorch_rule_model.py` implements a PyTorch transformer whose
weights are analytically constructed to be equivalent to the Tracr-compiled
RASP program — no JAX or Tracr dependency required.

- `d_model = VOCAB_SIZE + 4 = 28` (24 token dims + 4 period-4 position dims)
- 1 attention head, no MLP, no causal mask, all weights as fixed buffers
- Returns true residual stream: `h_out[q] = [e_{token[q]} + e_{token[(q+1)%4]}, e_{q%4}]`

### Cross-attention adapter (`YinyangCrossAttention`)

- Query = AR hidden state (128-dim), with sinusoidal pos enc added
- Key/Value = rule hidden state (28-dim)
- Causal mask, learnable scalar gate (init=1)
- Supports `indices_query` / `indices_key` for different temporal resolutions
- Per adapter (n_skip=2): ~14,337 params

### Data splits

| Split | Starters | Description |
|---|---|---|
| Pretrain-only | {0, 1} | AR pretrain only, not finetuned |
| Finetune-only | {6..15} | Finetuned only, not pretrained |
| Both | {2, 3, 4, 5} | Pretrained and finetuned |
| Neither | {17, 19, 20, 21} | Unseen — all individual tokens seen in training |

---

## Setup

```bash
pip install torch numpy peft
```

JAX/Tracr are **not required** — the rule model is implemented analytically in PyTorch.

---

## Training

### Full pipeline

```bash
python main.py
```

Runs: pretrain → finetune → yinyang → eval

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs_pretrain` | `200` | Epochs for AR pretraining |
| `--epochs_finetune` | `200` | Epochs for AR finetuning |
| `--epochs_yinyang` | `100` | Epochs for Yin-Yang adapter training |
| `--n_skip` | `2` | Inject rule every n_skip AR layers |
| `--d_model` | `128` | AR transformer hidden size |
| `--n_layers` | `4` | Number of AR transformer layers |
| `--n_heads` | `4` | Number of AR attention heads |
| `--ckpt_dir` | `checkpoints` | Directory to save checkpoints |

### Adapter only (skip pretrain/finetune)

If you already have `checkpoints/ar_transformer.pt`:

```bash
python training/train_yinyang.py \
    --ar_ckpt checkpoints/ar_transformer.pt \
    --n_skip 1 \
    --ckpt_name yinyang_skip1 \
    --no_lora \
    --epochs 100
```

n_skip ablation (run all three to compare):

```bash
for skip in 1 2 4; do
    python training/train_yinyang.py \
        --ar_ckpt checkpoints/ar_transformer.pt \
        --n_skip $skip --ckpt_name yinyang_skip$skip --no_lora --epochs 100
done
```

---

## Evaluation

```bash
python evaluate.py \
    --ar_ckpt checkpoints/ar_transformer.pt \
    --ft_ckpt checkpoints/ar_finetuned.pt \
    --verbose
```

Compares Pretrain / Finetune / Finetune+rule (skip=1,2,4) across all four data splits.

---

## Checkpoints

| File | Description |
|------|-------------|
| `ar_transformer.pt` | Pretrained AR model (starters 0–5) |
| `ar_finetuned.pt` | Finetuned AR model (starters 2–15) |
| `yinyang_skip1.pt` | Yin-Yang adapter, inject every layer (4 adapters) |
| `yinyang_skip2.pt` | Yin-Yang adapter, inject every 2 layers (2 adapters) |
| `yinyang_skip4.pt` | Yin-Yang adapter, inject every 4 layers (1 adapter) |

---

## Project Structure

```
rasp/
├── main.py                          # End-to-end pipeline runner
├── evaluate.py                      # Evaluation and results table
├── training/
│   ├── pretrain.py                  # Stage 1: AR transformer pretraining
│   ├── finetune.py                  # Stage 2: AR finetuning on B={2..15}
│   └── train_yinyang.py             # Stage 3: Yin-Yang adapter training
├── models/
│   ├── transformer.py               # Autoregressive transformer (~800k params)
│   ├── tracr_pytorch_rule_model.py  # Analytically-constructed Tracr-equivalent transformer
│   ├── rule_model.py                # Frozen rule model wrapper (rule_d_model=28)
│   └── yinyang_model.py             # Yin-Yang model + YinyangCrossAttention
├── rasp_program/
│   └── sequence_rule.py             # RASP program, FallbackRuleModel
├── data/
│   └── dataset.py                   # Dataset, dataloaders, starter splits
└── report.tex                       # Conference-style paper
```
