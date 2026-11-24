#!/usr/bin/env bash
# The five CP-bass stems that mirror the integer experiment, in one place.
#
# These commands previously lived only in the chat, so when the OneCycleLR
# off-by-one killed four of the five they had to be retyped from memory.
#
# Paths come from the environment so nothing is hardcoded to one cluster:
#
#   BASE_CKPT    pretrained CP base (keys A), the adapter's frozen host
#   TRAIN_DATA   finetune keys B
#   PRETRAIN_DATA  keys A, interleaved 50/50 (only the scratch stem uses it)
#   VAL_DATA     all 12 keys, for val_loss
#   UNSEEN_DATA  keys {6,8}, drives the unseen_acc checkpoint monitor
#   CKPT_DIR     default checkpoints_cp
#   LOG_DIR      default logs_cp
#
# Run one stem:    ./scripts/run_cp_stems.sh frozen
# Run several:     ./scripts/run_cp_stems.sh frozen lora trainar
# Run all five:    ./scripts/run_cp_stems.sh all
#
# Stems run sequentially (one GPU). Each logs to $LOG_DIR/<run_name>.log.

set -euo pipefail

: "${BASE_CKPT:?set BASE_CKPT to the pretrained CP base checkpoint}"
: "${TRAIN_DATA:?set TRAIN_DATA to the finetune (keys B) .pt}"
: "${VAL_DATA:?set VAL_DATA to the all-keys .pt}"
: "${UNSEEN_DATA:?set UNSEEN_DATA to the unseen (keys 6,8) .pt}"
PRETRAIN_DATA="${PRETRAIN_DATA:-}"
CKPT_DIR="${CKPT_DIR:-checkpoints_cp}"
LOG_DIR="${LOG_DIR:-logs_cp}"
MAX_STEPS="${MAX_STEPS:-10000}"
# One adapter per global layer. The integer experiment injects at every layer,
# so matching it here keeps the two experiments structurally comparable.
# model_size 1 has 12 global layers, so n_skip=1 gives 12 adapters (4.81M
# trainable, 4.12% of the host) against 3 at n_skip=4 (1.20M, 1.06%).
N_SKIP="${N_SKIP:-1}"
# Rule-model and masking options, so a run's configuration lives in the
# command rather than in a trainer default.
#   KEY_SELECT=first   the head attends exactly the slot holding beat 0,
#                      satisfying Tracr Appendix D.1, instead of averaging
#                      every phase-0 slot.
#   STRICT_CAUSAL=1    mask the adapter to j < t, so the rule's answer FOR
#                      beat t is withheld and the same-phase row at t-4 must
#                      be used instead.
KEY_SELECT="${KEY_SELECT:-phase0}"
#   RULE_RATE=bar      one adapter key per BAR, placed at the first beat each
#                      bar's root is known, instead of the same hint repeated
#                      at every beat. Also drops the zeroed index-0 row from
#                      the attention. 'beat' (default) is the original.
RULE_RATE="${RULE_RATE:-beat}"
STRICT_CAUSAL="${STRICT_CAUSAL:-0}"
#   SEED=<n>           adapter seed; pass a distinct RUN_SUFFIX per seed so
#                      the runs do not share a checkpoint directory.
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
#   GATE_INIT=<f>      adapter output-gate init. 0.0 (default) starts the
#                      adapter at zero contribution; 1.0 reproduces runs made
#                      before the flag existed.
#   FREEZE_BASE_EVAL=1 hold the frozen host in eval mode (dropout off). Only
#                      the frozen-host stems take it; it is rejected alongside
#                      --unfreeze_base.
#   LORA_RANK=<n>      rank for the lora stem. 16 matches the reference
#                      implementation and the hand-written launch blocks; this
#                      script previously hardcoded 8, so runs launched through
#                      it are not comparable to those on that axis.
GATE_INIT="${GATE_INIT:-0.0}"
FREEZE_BASE_EVAL="${FREEZE_BASE_EVAL:-0}"
LORA_RANK="${LORA_RANK:-16}"
#   BEATS_PER_CHORD=4  --cycle_unit bar data: the root is held for a bar, so
#                      the rule model holds each phase 4 steps and the probe
#                      scores downbeats only. 1 (default) is the per-beat data.
#   TRAIN_LENGTH=<n>   window in beats. A cycle is 4*BEATS_PER_CHORD beats, so
#                      use >= 32 when BEATS_PER_CHORD=4.
BEATS_PER_CHORD="${BEATS_PER_CHORD:-1}"
TRAIN_LENGTH="${TRAIN_LENGTH:-24}"
#   PROBE_PROMPT / PROBE_GEN  protocol for the in-training unseen_acc probe,
#                      which also drives checkpoint selection. Keep it equal to
#                      what the results table reports with. With one chord per
#                      bar a cycle is 16 beats, so a 1-beat prompt is a thin
#                      basis for the key: use 4 (one full bar) and 28.
PROBE_PROMPT="${PROBE_PROMPT:-1}"
PROBE_GEN="${PROBE_GEN:-23}"
# Window starts must land on a cycle boundary or the rule model, which reads
# phase from position, is misaligned for the whole window. The adapter trainer
# derives this itself; finetune_cp_bass has to be told.
SAMPLE_STEP=$(( 4 * BEATS_PER_CHORD ))
# model_size and the wandb project are fixed, not overridable: every run in
# this experiment is size 1 and belongs in the cp_bass project, and a stem
# that quietly differed on either would not be comparable to the rest.
RULE_OPTS=(--rule_key_select "$KEY_SELECT" --rule_rate "$RULE_RATE"
           --seed "$SEED" --gate_init "$GATE_INIT"
           --beats_per_chord "$BEATS_PER_CHORD" --train_length "$TRAIN_LENGTH"
           --probe_prompt_beats "$PROBE_PROMPT" --probe_gen_beats "$PROBE_GEN")
FROZEN_OPTS=()
[[ "$FREEZE_BASE_EVAL" == "1" ]] && FROZEN_OPTS+=(--freeze_base_eval)
[[ "$STRICT_CAUSAL" == "1" ]] && RULE_OPTS+=(--strict_causal)
# SHIFT_AUG=1: v0.42's pitch-shift augmentation, applied per window in the
# training loop, skipping any shift that would carry a song out of its own
# split (midi_adapter/shift_aug.py). Off by default so earlier runs reproduce.
SHIFT_AUG="${SHIFT_AUG:-0}"
AUG_OPTS=()
[[ "$SHIFT_AUG" == "1" ]] && AUG_OPTS+=(--shift_aug)
RULE_OPTS+=(${AUG_OPTS[@]+"${AUG_OPTS[@]}"})

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Which rule model the adapter reads. The default is the flagship root rule:
# the compiled MLP computes (key + OFFSETS[p]) % 12 and the adapter reads the
# answer subspace. RULE_MODE=chord switches to v6's CADENCE over the 24
# triads, where the symbol IS the chord and the adapter reads the next-chord
# block -- that needs a v6 dataset and BEATS_PER_CHORD=16.
RULE_MODE="${RULE_MODE:-compiled}"
RULE_VIEW="${RULE_VIEW:-root_next}"
CHORD_MINOR="${CHORD_MINOR:-harmonic}"
MODE_OPTS=(--rule_mode "$RULE_MODE" --rule_view "$RULE_VIEW")
[[ "$RULE_MODE" == chord ]] && MODE_OPTS+=(--chord_minor_pattern "$CHORD_MINOR")
# Every run still logs to cp_bass unless told otherwise.
WANDB_PROJECT="${WANDB_PROJECT:-cp_bass}"

COMMON=(
  --base_ckpt   "$BASE_CKPT"
  --train_data  "$TRAIN_DATA"
  --val_data    "$VAL_DATA"
  --unseen_data "$UNSEEN_DATA"
  --model_size 1 --batch_size 8 --n_skip "$N_SKIP"
  "${MODE_OPTS[@]}"
  "${RULE_OPTS[@]}"
  --max_steps "$MAX_STEPS" --val_check_interval 500
  --ckpt_dir "$CKPT_DIR"
  --no_tensorboard --wandb_project "$WANDB_PROJECT"
)

run () {            # run <run_name> <extra args...>
  local name="$1$RUN_SUFFIX"; shift
  echo "=== $name ==="
  python -m midi_adapter.train_cp_yinyang \
      "${COMMON[@]}" --run_name "$name" "$@" \
      2>&1 | tee "$LOG_DIR/$name.log"
}

stem () {
  case "$1" in
    # Frozen AR + adapter: the headline uni-directional FA setup.
    frozen)  run cp_yy_frozen "${FROZEN_OPTS[@]}" ;;
    # Same, with LoRA on the base Q/V.
    lora)    run cp_yy_lora    --lora_rank "$LORA_RANK" "${FROZEN_OPTS[@]}" ;;
    # Same, base fully trainable from the pretrained checkpoint.
    trainar) run cp_yy_trainar --unfreeze_base ;;
    # Base from scratch, trained jointly with the adapter on A+B interleaved.
    scratch)
      if [[ -z "$PRETRAIN_DATA" ]]; then
        echo "scratch needs PRETRAIN_DATA (keys A) for the 50/50 interleave" >&2
        exit 1
      fi
      python -m midi_adapter.train_cp_yinyang \
          --train_data "$TRAIN_DATA" --pretrain_data "$PRETRAIN_DATA" \
          --val_data "$VAL_DATA" --unseen_data "$UNSEEN_DATA" \
          --model_size 1 --batch_size 8 --n_skip "$N_SKIP" \
          "${MODE_OPTS[@]}" \
          "${RULE_OPTS[@]}" \
          --max_steps "$MAX_STEPS" --val_check_interval 500 \
          --ckpt_dir "$CKPT_DIR" --no_tensorboard \
          --wandb_project "$WANDB_PROJECT" \
          --unfreeze_base --run_name "cp_yy_scratch$RUN_SUFFIX" \
          2>&1 | tee "$LOG_DIR/cp_yy_scratch$RUN_SUFFIX.log"
      ;;
    # Baseline: plain finetuning on keys B, no adapter and no rule model.
    # Uses finetune_cp_bass, so none of the adapter options apply to it.
    finetune)
      python -m midi_adapter.finetune_cp_bass \
          --pretrain_ckpt "$BASE_CKPT" \
          --train_data "$TRAIN_DATA" --val_data "$VAL_DATA" \
          --model_size 1 --batch_size 8 --seed "$SEED" \
          --train_length "$TRAIN_LENGTH" --sample_step "$SAMPLE_STEP" \
          --max_steps "$MAX_STEPS" --val_check_interval 500 \
          --ckpt_dir "$CKPT_DIR" --run_name "cp_bass_ft$RUN_SUFFIX" ${AUG_OPTS[@]+"${AUG_OPTS[@]}"} \
          --no_tensorboard --wandb_project "$WANDB_PROJECT" \
          2>&1 | tee "$LOG_DIR/cp_bass_ft$RUN_SUFFIX.log"
      ;;
    # No input to the rule model: the host's pre-block state is encoded into
    # the rule model's token subspace. MLP encoder + aux CE on the current pc,
    # the variant that reached 1.000 on the integer experiment.
    noinput) run cp_yy_noinput_mlp_aux1 \
                 --rule_input ar_state --rule_enc_type mlp --rule_enc_hidden 64 \
                 --enc_loss_weight 1.0 "${FROZEN_OPTS[@]}" ;;
    *) echo "unknown stem: $1  (finetune frozen lora trainar scratch noinput all)" >&2
       exit 1 ;;
  esac
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <finetune|frozen|lora|trainar|scratch|noinput|all> ..." >&2
  exit 1
fi

if [[ "$1" == "all" ]]; then
  set -- finetune frozen lora trainar scratch noinput
fi

for s in "$@"; do stem "$s"; done
echo "done: $*"
