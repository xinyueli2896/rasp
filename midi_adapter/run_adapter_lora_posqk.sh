#!/usr/bin/env bash
# =============================================================================
#  Train the FIXED adapter WITH LoRA (positional-QK alignment + LoRA-16 on the
#  base Q/V projections) on both data configurations, then evaluate each on
#  both test domains.
#
#    run 1: direct-only data  → pop909_ivvi_w1_adapter_lora16_posqk_direct
#    run 2: mixed data        → pop909_ivvi_w1_adapter_lora16_posqk_mixed
#
#  This is the closest reconstruction of the integer-experiment recipe
#  (models/yinyang_model.py), which used BOTH:
#    * purely positional Q/K scaled x20  → alignment hardcoded, not learned
#    * LoRA rank-16 on the frozen base   → base can co-adapt to the correction
#  The base remains otherwise frozen; only adapters + LoRA A/B train.
#
#  Usage:
#    bash midi_adapter/run_adapter_lora_posqk.sh            # both runs + evals
#    bash midi_adapter/run_adapter_lora_posqk.sh direct     # only the direct run
#    bash midi_adapter/run_adapter_lora_posqk.sh mixed      # only the mixed run
# =============================================================================

set -euo pipefail

RASP_REPO=/l/users/xinyue.li/rasp
D=/l/users/xinyue.li/data/pop909_ivvi_w1
BASE=checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
LORA_RANK=16

COMMON_TRAIN="--base_ckpt $BASE \
    --approach chord --n_skip 1 --paired_chord_seq --positional_qk \
    --lora_rank $LORA_RANK \
    --chords_per_bar 2 --model_size 1 --adapter_rank 256 --batch_size 8 \
    --max_steps 40000"

# NOTE: --lora_rank and --positional_qk MUST be repeated at eval time, or the
# checkpoint's LoRA weights are silently dropped by the strict=False load.
COMMON_EVAL="--base_ckpt $BASE \
    --approach chord --n_skip 1 --paired_chord_seq --chords_per_bar 2 \
    --positional_qk --lora_rank $LORA_RANK \
    --n_prompt_beats 16 --temperature 0 --save_n_per_key 3"

cd "$RASP_REPO"
mkdir -p eval_logs_grid
WHICH="${1:-both}"

log() { echo -e "\n════════════════════════════════════════════\n▶ $*\n════════════════════════════════════════════"; }

best_ckpt() {   # best_ckpt <run_name>  → path of lowest-val_loss ckpt
    ls "checkpoints/$1/"*.by_val_loss.*.ckpt 2>/dev/null \
        | sort -t= -k3 -g | head -1
}

# ─── Run 1: direct-only ─────────────────────────────────────────────────────
if [ "$WHICH" = both ] || [ "$WHICH" = direct ]; then
    RUN=pop909_ivvi_w1_adapter_lora16_posqk_direct
    if [ -n "$(best_ckpt $RUN)" ]; then
        log "$RUN — checkpoints exist, skipping training"
    else
        log "$RUN — training (direct-only data, LoRA-$LORA_RANK + positional QK)"
        python -m midi_adapter.train_cp_yinyang $COMMON_TRAIN \
            --train_data  "$D/direct_train_seenkeys.pt" \
            --val_data    "$D/direct_val_seenkeys.pt" \
            --unseen_data "$D/direct_val_unseenkeys.pt" \
            --run_name "$RUN"
    fi

    CKPT=$(best_ckpt $RUN)
    log "$RUN — evaluating $CKPT"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL \
        --adapter_ckpt "$CKPT" \
        --seen_data "$D/val_all_keys_seenkeys.pt" --unseen_data "$D/val_all_keys_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_lora_posqk_direct_orch/" \
        2>&1 | tee "eval_logs_grid/adapter_lora_posqk_direct_orch.log"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL \
        --adapter_ckpt "$CKPT" \
        --seen_data "$D/direct_val_seenkeys.pt" --unseen_data "$D/direct_val_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_lora_posqk_direct_direct/" \
        2>&1 | tee "eval_logs_grid/adapter_lora_posqk_direct_direct.log"
fi

# ─── Run 2: mixed ───────────────────────────────────────────────────────────
if [ "$WHICH" = both ] || [ "$WHICH" = mixed ]; then
    RUN=pop909_ivvi_w1_adapter_lora16_posqk_mixed
    if [ -n "$(best_ckpt $RUN)" ]; then
        log "$RUN — checkpoints exist, skipping training"
    else
        log "$RUN — training (mixed data, LoRA-$LORA_RANK + positional QK)"
        python -m midi_adapter.train_cp_yinyang $COMMON_TRAIN \
            --train_data    "$D/train_all_keys_seenkeys.pt" \
            --pretrain_data "$D/direct_train_seenkeys.pt" \
            --val_data      "$D/val_all_keys_seenkeys.pt" \
            --unseen_data   "$D/val_all_keys_unseenkeys.pt" \
            --run_name "$RUN"
    fi

    CKPT=$(best_ckpt $RUN)
    log "$RUN — evaluating $CKPT"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL \
        --adapter_ckpt "$CKPT" \
        --seen_data "$D/val_all_keys_seenkeys.pt" --unseen_data "$D/val_all_keys_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_lora_posqk_mixed_orch/" \
        2>&1 | tee "eval_logs_grid/adapter_lora_posqk_mixed_orch.log"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL \
        --adapter_ckpt "$CKPT" \
        --seen_data "$D/direct_val_seenkeys.pt" --unseen_data "$D/direct_val_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_lora_posqk_mixed_direct/" \
        2>&1 | tee "eval_logs_grid/adapter_lora_posqk_mixed_direct.log"
fi

log "Done. Logs in eval_logs_grid/adapter_lora_posqk_*.log"
