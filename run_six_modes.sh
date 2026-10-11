#!/usr/bin/env bash
#
# Launch all integer-experiment modes on an 8-GPU node.
#
#   1. AR pretrain on A                                 (1 run, no --seed flag)
#   2. finetune without adapter on B                    (1 run, no --seed flag)
#   3. adapter + pretrained AR frozen                   (5 seeds)
#   4. adapter + pretrained AR trainable                (5 seeds)
#   5. adapter + LoRA on pretrained AR                  (5 seeds)
#   6. adapter + AR from scratch, jointly, on A u B     (5 seeds)
#   7. adapter + learned rule-input encoder, AR frozen  (5 seeds)
#
# Dependency structure: modes 2-5 and 7 need checkpoints/ar_transformer.pt from
# mode 1.  Mode 6 trains its own AR from scratch and depends on nothing, so it
# starts immediately alongside the pretrain and keeps running underneath the
# rest -- the gate below waits on the pretrain PID only, never on mode 6.
#
# GPU assignment is a round-robin counter over N_GPUS, so every job is launched
# concurrently and the load is spread evenly.  These are ~0.8M-parameter models,
# so several per GPU is free.
#
# Env overrides:
#   N_GPUS            visible GPUs to spread across (default 8)
#   OMP_NUM_THREADS   per-process thread cap (default 4; prevents 26 torch
#                     processes from oversubscribing the node's cores)
#   ENC_LOSS_WEIGHT   mode 7 auxiliary encoder CE weight (default 0.0 = the
#                     unsupervised CP default; the CP README recommends 1.0
#                     for the supervised variant)
#   ENC_TYPE          mode 7 encoder: embedding (default, warm-started from the
#                     frozen one-hot W_E) or token_mlp (xavier, not warm-started)
#   SKIP_LORA=1       skip mode 5 when the peft package is unavailable
#   SKIP_WAVE1=1      modes 1 and 6 already ran; launch only the second wave
#   NO_EVAL=1         skip the final evaluation table

set -u

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs checkpoints

N_GPUS="${N_GPUS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

AR=checkpoints/ar_transformer.pt
COMMON="--n_skip 1 --epochs 400 --n_cycles 8 --n_seqs_per_starter 200"
ENC_LOSS_WEIGHT="${ENC_LOSS_WEIGHT:-0.0}"
ENC_TYPE="${ENC_TYPE:-embedding}"
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_WAVE1="${SKIP_WAVE1:-0}"
NO_EVAL="${NO_EVAL:-0}"

# --- preflight: mode 5 hard-errors inside build_yinyang_model without peft ---
if [ "$SKIP_LORA" != "1" ] && ! python -c "import peft" 2>/dev/null; then
    echo "ERROR: mode 5 (LoRA) needs the 'peft' package, which is not importable." >&2
    echo "       Install it:   python -m pip install peft" >&2
    echo "       Or skip it:   SKIP_LORA=1 $0" >&2
    exit 1
fi

# Round-robin GPU counter.  Shell functions do not run in a subshell, so the
# increment persists across calls.
GPU=0
LAST_PID=""
run () {   # run <logname> <script> <args...>
    local log=$1
    shift
    local gpu=$((GPU % N_GPUS))
    GPU=$((GPU + 1))
    CUDA_VISIBLE_DEVICES="$gpu" python "$@" > "logs/$log.log" 2>&1 &
    LAST_PID=$!
    printf '  gpu%-2s %-22s pid=%s\n' "$gpu" "$log" "$LAST_PID"
}

if [ "$SKIP_WAVE1" = "1" ]; then
    echo "SKIP_WAVE1=1: modes 1 and 6 assumed already done."
    if [ ! -f "$AR" ]; then
        echo "ERROR: $AR missing -- cannot skip wave 1." >&2
        exit 1
    fi
    for S in 0 1 2 3 4; do
        [ -f "checkpoints/yinyang_scratch_seed$S.pt" ] \
            || echo "  WARNING: checkpoints/yinyang_scratch_seed$S.pt missing (mode 6 seed $S)" >&2
    done
    # Start the round-robin where wave 1 left off so the load stays spread.
    GPU=6
else
    echo "=============================================================="
    echo "Wave 1: AR pretrain (mode 1)  ||  scratch-joint (mode 6)"
    echo "=============================================================="

    run pretrain training/pretrain.py \
        --epochs 200 --lr 3e-4 --n_cycles 8 --n_seqs_per_starter 200 \
        --ckpt_name ar_transformer
    PRETRAIN_PID=$LAST_PID

    for S in 0 1 2 3 4; do
        # --ar_ckpt '' is what forces from-scratch: _build() loads a checkpoint
        # only when the path is non-empty AND exists, and $AR will exist once
        # mode 1 ends.
        run "scratch_seed$S" training/train_yinyang.py \
            --seed "$S" --ckpt_name "yinyang_scratch_seed$S" \
            --joint --train_ar --ar_ckpt '' --no_lora $COMMON
    done

    # --- gate: wave 2 needs ONLY the pretrained checkpoint, not mode 6 ------
    echo
    echo "Waiting on pretrain (pid $PRETRAIN_PID); mode 6 keeps running ..."
    wait "$PRETRAIN_PID"
    PRE_RC=$?
    if [ "$PRE_RC" -ne 0 ] || [ ! -f "$AR" ]; then
        echo "ERROR: pretrain exited $PRE_RC / $AR missing. See logs/pretrain.log." >&2
        echo "       Letting mode 6 finish, then aborting." >&2
        wait
        exit 1
    fi
    echo "Pretrain done -> $AR"
fi

echo
echo "=============================================================="
echo "Wave 2: modes 2-5 and 7, overlapping mode 6"
echo "=============================================================="

# (2) finetune WITHOUT adapter = continue pretraining on B
run finetune training/finetune.py \
    --ar_ckpt "$AR" --epochs 200 --lr 1e-4 \
    --n_cycles 8 --n_seqs_per_starter 200 --ckpt_name ar_finetuned

# (3) adapter + pretrained AR frozen
for S in 0 1 2 3 4; do
    run "frozen_seed$S" training/train_yinyang.py \
        --seed "$S" --ckpt_name "yinyang_frozen_seed$S" \
        --ar_ckpt "$AR" --no_lora $COMMON
done

# (4) adapter + pretrained AR trainable
for S in 0 1 2 3 4; do
    run "trainar_seed$S" training/train_yinyang.py \
        --seed "$S" --ckpt_name "yinyang_trainar_seed$S" \
        --ar_ckpt "$AR" --train_ar --no_lora $COMMON
done

# (5) adapter + LoRA on pretrained AR
if [ "$SKIP_LORA" != "1" ]; then
    for S in 0 1 2 3 4; do
        run "lora_seed$S" training/train_yinyang.py \
            --seed "$S" --ckpt_name "yinyang_lora_seed$S" \
            --ar_ckpt "$AR" --use_lora --lora_rank 16 $COMMON
    done
fi

# (7) adapter + learned rule-input encoder, AR frozen  (CP "enc_inject")
for S in 0 1 2 3 4; do
    run "enc_seed$S" training/train_yinyang.py \
        --seed "$S" --ckpt_name "yinyang_enc_seed$S" \
        --ar_ckpt "$AR" --no_lora \
        --encoder_injected --encoder_type "$ENC_TYPE" \
        --enc_loss_weight "$ENC_LOSS_WEIGHT" $COMMON
done

echo
echo "All jobs launched. Waiting for completion ..."
wait
echo "All training finished."

if [ "$NO_EVAL" = "1" ]; then
    exit 0
fi

STEMS="yinyang_frozen yinyang_trainar yinyang_scratch yinyang_enc"
[ "$SKIP_LORA" != "1" ] && STEMS="$STEMS yinyang_lora"

echo
echo "=============================================================="
echo "Evaluation"
echo "=============================================================="
python evaluate.py \
    --ar_ckpt "$AR" --ft_ckpt checkpoints/ar_finetuned.pt \
    --adapter_stems $STEMS \
    --prompt_len 1 --n_cycles 8 2>&1 | tee logs/results.txt
