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
- Returns unambiguous rule signal: `h_out[q] = [e_{predicted_next[q]}, e_{q%4}]`

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

## Bass CP Experiment

A parallel experiment that tests the same Yin-Yang rule-adapter idea on symbolic
music: a CP transformer trained on bass lines that follow the I–IV–V–I cycle.

**The rule:** at beat `t`, the bass pitch class is `(key + OFFSETS[t % 4]) % 12`
where `key` is the starting pitch class and `OFFSETS = [0, 5, 7, 0]`.
Example for `key=0` (C): `C, F, G, C, C, F, G, C, ...`

### Key splits

| Split | Keys | Description |
|---|---|---|
| Pretrain-only | {5, 9} | Seen only during base CP pretraining |
| Finetune-only | {1, 2, 3, 4, 7, 10, 11} | Seen only during adapter finetuning |
| Both seen | {0} | Seen in both pretrain and finetune |
| Unseen | {6, 8} | Never seen during any training — generalization test |

### Rule model

`models/bass_tracr_rule_model.py` — analytical, zero parameters.
- `d_model = 12 + 4 = 16` (12 pitch-class dims + 4 period-4 position dims)
- `rule_hidden[t]` encodes the **current** pitch class at `t`:
  token dims = `one_hot(key + OFFSETS[t%4])`, position dims = `one_hot(t%4)`
- CURRENT encoding is used because the causal mask blocks `rule_hidden[t+1]`.
  The adapter learns to attend `rule_hidden[0]` (= `embed(key)`, always accessible)
  and uses query position encoding to derive the next note — routing that
  generalises to unseen keys. NEXT encoding causes a 0.25 accuracy collapse
  where the model fixates on `rule_hidden[1]` (= `embed(key+7)`) for all positions.

### Adapter variants

| Flag | Description |
|---|---|
| *(none)* | Standard: frozen analytical rule model |
| `--encoder_injected` | Learned `nn.Embedding(12, 16)` replaces the one-hot `W_E` lookup |
| `--bidirectional` | No rule model input; learned `Linear(d_model, 16)` projects AR states to rule space |

### Full training pipeline

**Step 1 — generate datasets**

```bash
# Base pretrain data  (keys 0, 5, 9)
python -m midi_adapter.generate_synthetic_bass \
    --n_songs 3000 --n_bars 32 --seed 42 \
    --keys 0 5 9 \
    --out_dir data/bass_pretrain --out_pt data/bass_pretrain_cp4

# Adapter finetune data  (keys 0–4, 7, 10, 11)
python -m midi_adapter.generate_synthetic_bass \
    --n_songs 3000 --n_bars 32 --seed 42 \
    --keys 0 1 2 3 4 7 10 11 \
    --out_dir data/bass_finetune --out_pt data/bass_finetune_cp4

# Validation / unseen-key eval  (all 12 keys)
python -m midi_adapter.generate_synthetic_bass \
    --n_songs 1000 --n_bars 32 --seed 99 \
    --out_dir data/bass_all --out_pt data/bass_all_cp4
```

**Step 2 — pretrain base CP transformer**  (max_steps=200,000)

```bash
python -m midi_adapter.pretrain_cp_bass \
    --train_data data/bass_pretrain_cp4.pt \
    --val_data   data/bass_pretrain_cp4.pt \
    --model_size 1 --batch_size 8 \
    --ckpt_dir   ckpt/cp_bass_pretrained
```

Checkpoint saved to `ckpt/cp_bass_pretrained/last.ckpt`.

**Step 3 — train the adapter**  (max_steps=100,000)

Standard adapter:
```bash
python -m midi_adapter.train_cp_yinyang \
    --base_ckpt  ckpt/cp_bass_pretrained/last.ckpt \
    --train_data data/bass_finetune_cp4.pt \
    --val_data   data/bass_all_cp4.pt \
    --model_size 1 --batch_size 8 \
    --adapter_rank 256 --n_skip 4 \
    --ckpt_dir   checkpoints
```

Encoder-injected (learned pitch-class embedding):
```bash
python -m midi_adapter.train_cp_yinyang \
    --base_ckpt  ckpt/cp_bass_pretrained/last.ckpt \
    --train_data data/bass_finetune_cp4.pt \
    --val_data   data/bass_all_cp4.pt \
    --model_size 1 --batch_size 8 \
    --adapter_rank 256 --n_skip 4 \
    --encoder_injected \
    --ckpt_dir   checkpoints
```

Bidirectional (no rule model input):
```bash
python -m midi_adapter.train_cp_yinyang \
    --base_ckpt  ckpt/cp_bass_pretrained/last.ckpt \
    --train_data data/bass_finetune_cp4.pt \
    --val_data   data/bass_all_cp4.pt \
    --model_size 1 --batch_size 8 \
    --adapter_rank 256 --n_skip 4 \
    --bidirectional \
    --ckpt_dir   checkpoints
```

Joint training with pretrain + finetune data interleaved:
```bash
python -m midi_adapter.train_cp_yinyang \
    --base_ckpt    ckpt/cp_bass_pretrained/last.ckpt \
    --train_data   data/bass_finetune_cp4.pt \
    --pretrain_data data/bass_pretrain_cp4.pt \
    --val_data     data/bass_all_cp4.pt \
    --model_size 1 --batch_size 8 \
    --adapter_rank 256 --n_skip 4 \
    --ckpt_dir     checkpoints
```

**Step 4 — evaluate**

```bash
python -m midi_adapter.evaluate_cp_yinyang \
    --base_ckpt    ckpt/cp_bass_pretrained/last.ckpt \
    --adapter_ckpt checkpoints/<run_name>/best.pt \
    --model_size 1 \
    --n_trials 4 --temperature 1.0
```

Reports per-key accuracy broken down by split (pretrain-only, finetune-only,
both-seen, unseen). The key metric is unseen-key accuracy `{6, 8}`.

### Bass project structure

```
rasp/
├── models/
│   ├── bass_tracr_rule_model.py     # Analytical rule model (mod 12, 16-dim hidden)
│   └── tracr_pytorch_rule_model.py  # Integer rule model (mod 24, 28-dim hidden)
└── midi_adapter/
    ├── cp_yinyang.py                # CPYinyangTransformer + CPYinyangCrossAttention
    ├── train_cp_yinyang.py          # Adapter training (Lightning)
    ├── evaluate_cp_yinyang.py       # Accuracy evaluation across key splits
    ├── pretrain_cp_bass.py          # Base CP transformer pretraining
    ├── finetune_cp_bass.py          # Base CP transformer finetuning
    ├── generate_synthetic_bass.py   # Synthetic bass dataset generation
    ├── evaluate_cp_bass.py          # Base model evaluation
    ├── chord_tokenizer.py           # CP token encoding/decoding
    └── infer_cp_bass.py             # Single-key inference helper
```

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
