#!/usr/bin/env bash
# Runs the original experiment pipeline, skipping checkpoints that already exist.
#
# Already done (skipped automatically):
#   Pretrain {0..6}, Finetune x3, adapter_2to16 seeds 0-4
#
# Remaining:
#   adapter_0to16 seeds 0-4, adapter_0to16_plus seeds 0-4, evaluation x3

set -e

BASE="$(cd "$(dirname "$0")" && pwd)"
DATASET="$BASE/data/dataset.py"
AR_CKPT="checkpoints/ar_pretrain_0to6.pt"
EPOCHS=400
N_SEEDS=5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
patch_adapter() {
    python3 - "$DATASET" "$1" <<'EOF'
import sys
path, expr = sys.argv[1], sys.argv[2]
with open(path) as f:
    lines = f.readlines()
out = []
for line in lines:
    s = line.strip()
    if s.startswith('ADAPTER_TRAIN_STARTERS') and '=' in s:
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
trap revert EXIT

ckpt_exists() {
    [ -f "$BASE/checkpoints/${1}.pt" ]
}

run_seed() {
    local stem="$1"
    local seed="$2"
    local name="${stem}_seed${seed}"
    if ckpt_exists "$name"; then
        echo "  skip $name (already exists)"
        return
    fi
    echo ""
    echo "----------------------------------------------------------------------"
    echo "  $name"
    echo "----------------------------------------------------------------------"
    python3 training/train_yinyang.py \
        --seed      "$seed" \
        --ckpt_name "$name" \
        --n_skip    1 \
        --no_lora \
        --epochs    "$EPOCHS" \
        --ar_ckpt   "$AR_CKPT"
}

# ---------------------------------------------------------------------------
# adapter_2to16  (seeds 0-4)
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "ADAPTER  starters={2..16}"
echo "======================================================================"
patch_adapter 'list(range(2, 17))'
for seed in 0 1 2 3 4; do
    run_seed "adapter_2to16" "$seed"
done

# ---------------------------------------------------------------------------
# adapter_0to16  (seeds 0-4)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "ADAPTER  starters={0..16}"
echo "======================================================================"
patch_adapter 'list(range(0, 17))'
for seed in 0 1 2 3 4; do
    run_seed "adapter_0to16" "$seed"
done

# ---------------------------------------------------------------------------
# adapter_0to16_plus  (seeds 0-4)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "ADAPTER  starters={0..16,18,22,23}"
echo "======================================================================"
patch_adapter 'list(range(0, 17)) + [18, 22, 23]'
for seed in 0 1 2 3 4; do
    run_seed "adapter_0to16_plus" "$seed"
done

revert
trap - EXIT

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "EVALUATION"
echo "======================================================================"

for ft_ckpt in ar_finetuned_2to16 ar_finetuned_0to16 ar_finetuned_0to16_plus; do
    echo ""
    echo "--- ft=$ft_ckpt ---"
    python3 evaluate.py \
        --ar_ckpt       "$AR_CKPT" \
        --ft_ckpt       "checkpoints/${ft_ckpt}.pt" \
        --adapter_stems adapter_2to16 adapter_0to16 adapter_0to16_plus \
        --verbose
done

echo ""
echo "=== ALL DONE ==="
