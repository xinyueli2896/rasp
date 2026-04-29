#!/usr/bin/env bash
# Adapter training with trainable AR model — two variants × 3 dataset configs × 5 seeds
#
# Variant A: AR initialised from pretrained checkpoint (checkpoints/ar_pretrain_0to6.pt)
# Variant B: AR trained from scratch (no checkpoint)
#
# Dataset configs:
#   2to16      : starters {2..16}
#   0to16      : starters {0..16}
#   0to16_plus : starters {0..16, 18, 22, 23}

set -e

BASE="$(cd "$(dirname "$0")" && pwd)"
DATASET="$BASE/data/dataset.py"
AR_CKPT="checkpoints/ar_pretrain_0to6.pt"
EPOCHS=400
N_SEEDS=5

# ---------------------------------------------------------------------------
# Helper: patch ADAPTER_TRAIN_STARTERS in dataset.py
# ---------------------------------------------------------------------------
patch_adapter() {
    python3 - "$DATASET" "$1" <<'EOF'
import sys
path, expr = sys.argv[1], sys.argv[2]
with open(path) as f:
    lines = f.readlines()
out = []
for line in lines:
    if line.strip().startswith('ADAPTER_TRAIN_STARTERS') and '=' in line:
        out.append(f'ADAPTER_TRAIN_STARTERS = {expr}\n')
    else:
        out.append(line)
with open(path, 'w') as f:
    f.writelines(out)
EOF
}

revert() {
    patch_adapter 'FINETUNE_STARTERS'
    echo "[dataset.py reverted]"
}

# Revert on exit (including errors)
trap revert EXIT

# ---------------------------------------------------------------------------
# Dataset configs: (expression, stem_suffix, label)
# ---------------------------------------------------------------------------
declare -a EXPRS=( 'list(range(2, 17))' 'list(range(0, 17))' 'list(range(0, 17)) + [18, 22, 23]' )
declare -a STEMS=( '2to16'              '0to16'              '0to16_plus' )
declare -a LABELS=( '{2..16}'           '{0..16}'            '{0..16,18,22,23}' )

# ---------------------------------------------------------------------------
# Variant A: trainable AR, initialised from pretrained checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "VARIANT A: trainable AR + adapter, init from $AR_CKPT"
echo "======================================================================"

for i in 0 1 2; do
    expr="${EXPRS[$i]}"
    stem="adapter_train_ar_${STEMS[$i]}"
    label="${LABELS[$i]}"

    echo ""
    echo "--- Config: starters=$label  stem=$stem ---"
    patch_adapter "$expr"

    python3 training/train_yinyang_multirun.py \
        --n_seeds  "$N_SEEDS" \
        --ckpt_stem "$stem" \
        -- \
        --n_skip   1 \
        --no_lora \
        --train_ar \
        --ar_ckpt  "$AR_CKPT" \
        --epochs   "$EPOCHS"
done

# ---------------------------------------------------------------------------
# Variant B: trainable AR, trained from scratch (no checkpoint)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "VARIANT B: trainable AR + adapter, trained from scratch"
echo "======================================================================"

for i in 0 1 2; do
    expr="${EXPRS[$i]}"
    stem="adapter_train_ar_scratch_${STEMS[$i]}"
    label="${LABELS[$i]}"

    echo ""
    echo "--- Config: starters=$label  stem=$stem ---"
    patch_adapter "$expr"

    python3 training/train_yinyang_multirun.py \
        --n_seeds  "$N_SEEDS" \
        --ckpt_stem "$stem" \
        -- \
        --n_skip   1 \
        --no_lora \
        --train_ar \
        --epochs   "$EPOCHS"
        # No --ar_ckpt: AR trains from random initialisation
done

echo ""
echo "=== ALL TRAIN_AR EXPERIMENTS COMPLETE ==="
