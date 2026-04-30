#!/usr/bin/env bash
# Overlap experiments: vary A∩B while fixing |A| and |A∪B|.
#
# For each config, runs the full pipeline:
#   1. Pretrain AR on A
#   2. Finetune AR on B  (from pretrained ckpt)
#   3. Train adapter x5 seeds on B (frozen pretrained AR)
#
# Series 1: |A|=6,  |A∪B|=16
# Series 2: |A|=10, |A∪B|=20
#
# Test starters are always {17,19,20,21}.

set -e

BASE="$(cd "$(dirname "$0")" && pwd)"
DATASET="$BASE/data/dataset.py"
PRETRAIN_EPOCHS=200
FINETUNE_EPOCHS=200
ADAPTER_EPOCHS=400
N_SEEDS=5

# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------
patch_dataset() {
    # Usage: patch_dataset <ar_expr> <ft_expr>
    python3 - "$DATASET" "$1" "$2" <<'EOF'
import sys
path, ar_expr, ft_expr = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    lines = f.readlines()
out = []
for line in lines:
    s = line.strip()
    if s.startswith('AR_TRAIN_STARTERS') and '=' in s:
        out.append(f'AR_TRAIN_STARTERS      = {ar_expr}\n')
    elif s.startswith('FINETUNE_STARTERS') and '=' in s:
        out.append(f'FINETUNE_STARTERS      = {ft_expr}\n')
    elif s.startswith('ADAPTER_TRAIN_STARTERS') and '=' in s:
        out.append(f'ADAPTER_TRAIN_STARTERS = {ft_expr}\n')
    else:
        out.append(line)
with open(path, 'w') as f:
    f.writelines(out)
EOF
}

revert() {
    patch_dataset 'list(range(6))' 'list(range(2, 16))'
    echo "[dataset.py reverted]"
}
trap revert EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ckpt_exists() { [ -f "checkpoints/${1}.pt" ]; }

all_seeds_exist() {
    local stem="$1"
    local n="$2"
    for i in $(seq 0 $((n - 1))); do
        [ -f "checkpoints/${stem}_seed${i}.pt" ] || return 1
    done
    return 0
}

# ---------------------------------------------------------------------------
# Run one full config — skips steps whose checkpoints already exist
# Usage: run_config <stem> <ar_expr> <ft_expr> <label>
# ---------------------------------------------------------------------------
run_config() {
    local stem="$1"
    local ar_expr="$2"
    local ft_expr="$3"
    local label="$4"

    echo ""
    echo "======================================================================"
    echo "CONFIG: $label  (stem=$stem)"
    echo "  A = $ar_expr"
    echo "  B = $ft_expr"
    echo "======================================================================"

    local pt_ckpt="checkpoints/${stem}_pretrain.pt"
    local ft_ckpt="checkpoints/${stem}_finetune.pt"

    # 1. Pretrain AR on A
    if ckpt_exists "${stem}_pretrain"; then
        echo "[SKIP] Step 1: Pretrain  (${stem}_pretrain.pt exists)"
    else
        echo "--- Step 1: Pretrain ---"
        patch_dataset "$ar_expr" "$ft_expr"
        python3 training/pretrain.py \
            --epochs "$PRETRAIN_EPOCHS" \
            --ckpt_name "${stem}_pretrain"
    fi

    # 2. Finetune AR on B
    if ckpt_exists "${stem}_finetune"; then
        echo "[SKIP] Step 2: Finetune  (${stem}_finetune.pt exists)"
    else
        echo "--- Step 2: Finetune ---"
        patch_dataset "$ar_expr" "$ft_expr"
        python3 training/finetune.py \
            --epochs "$FINETUNE_EPOCHS" \
            --ar_ckpt "$pt_ckpt" \
            --ckpt_name "${stem}_finetune"
    fi

    # 3. Adapter x5 seeds (frozen pretrained AR)
    if all_seeds_exist "${stem}_adapter" "$N_SEEDS"; then
        echo "[SKIP] Step 3: Adapter   (all ${N_SEEDS} seeds exist)"
    else
        echo "--- Step 3: Adapter (x${N_SEEDS} seeds, frozen pretrained AR) ---"
        patch_dataset "$ar_expr" "$ft_expr"
        python3 training/train_yinyang_multirun.py \
            --n_seeds  "$N_SEEDS" \
            --ckpt_stem "${stem}_adapter" \
            -- \
            --n_skip   1 \
            --no_lora \
            --epochs   "$ADAPTER_EPOCHS" \
            --ar_ckpt  "$pt_ckpt"
    fi
}

# ---------------------------------------------------------------------------
# Series 1: |A|=6, |A∪B|=16
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "SERIES 1:  B={6..15} fixed, |A∪B|=20 fixed, A grows with overlap"
echo "  A\B = {0,1,2,3,4,5,16,18,22,23} (fixed, 10 starters)"
echo "  A∩B grows from {} to {6,7,...,11} as overlap increases"
echo "  |A| grows from 10 to 16; |A∪B| = 20 stays constant"
echo "  Constraint satisfied: A covers tokens 0-4; B covers tokens 17,19-22"
echo "######################################################################"

# B = {6..15} fixed.
# A = {0,1,2,3,4,5,16,18,22,23} (A\B, fixed) ∪ first |A∩B| elements of B.
# |A∪B| = |A\B| + |B| = 10 + 10 = 20 always.

run_config "s1_ov0" \
    "[0,1,2,3,4,5,16,18,22,23]" \
    "[6,7,8,9,10,11,12,13,14,15]" \
    "S1 overlap=0  A={0..5,16,18,22,23}  B={6..15}"

run_config "s1_ov2" \
    "[0,1,2,3,4,5,16,18,22,23,6,7]" \
    "[6,7,8,9,10,11,12,13,14,15]" \
    "S1 overlap=2  A={0..5,16,18,22,23,6,7}  B={6..15}"

run_config "s1_ov4" \
    "[0,1,2,3,4,5,16,18,22,23,6,7,8,9]" \
    "[6,7,8,9,10,11,12,13,14,15]" \
    "S1 overlap=4  A={0..5,16,18,22,23,6..9}  B={6..15}"

run_config "s1_ov6" \
    "[0,1,2,3,4,5,16,18,22,23,6,7,8,9,10,11]" \
    "[6,7,8,9,10,11,12,13,14,15]" \
    "S1 overlap=6  A={0..5,16,18,22,23,6..11}  B={6..15}"

# ---------------------------------------------------------------------------
# Series 2: |A|=10, |A∪B|=20
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "SERIES 2:  |A|=10  |A∪B|=20"
echo "######################################################################"

run_config "s2_ov0" \
    "[0,1,2,3,4,5,16,18,22,23]" \
    "[6,7,8,9,10,11,12,13,14,15]" \
    "S2 overlap=0"

run_config "s2_ov2" \
    "[0,1,2,3,4,5,16,18,6,7]" \
    "[6,7,8,9,10,11,12,13,14,15,22,23]" \
    "S2 overlap=2"

run_config "s2_ov4" \
    "[0,1,2,3,4,5,6,7,8,9]" \
    "[6,7,8,9,10,11,12,13,14,15,16,18,22,23]" \
    "S2 overlap=4"

run_config "s2_ov6" \
    "[0,1,2,3,6,7,8,9,10,11]" \
    "[6,7,8,9,10,11,12,13,14,15,16,18,22,23,4,5]" \
    "S2 overlap=6"

echo ""
echo "=== ALL OVERLAP EXPERIMENTS COMPLETE ==="
