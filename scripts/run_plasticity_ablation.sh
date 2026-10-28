#!/usr/bin/env bash
# How much host plasticity does it take before the rule channel stops being
# used?
#
# Three points are already known: a frozen host reaches 1.000 on held-out keys,
# LoRA rank 16 reaches 0.993, and a fully trainable host reaches 0.040. The
# frozen host scores 0.000 when fed a wrong rule input; the trainable one still
# scores 0.865, and on key 0 corrupting the rule changes nothing at all. So the
# trainable host solved the eight finetune keys by itself -- 8 keys x 4 phases
# is 32 facts -- and never made the channel load-bearing.
#
# That suggests capacity, not frozen-vs-not, is the variable. This sweeps LoRA
# rank between the two extremes and measures, at each rank:
#
#   unseen accuracy        does it still generalise?
#   acc with a wrong rule  does it still depend on the channel?
#
# The prediction is that both hold up to some rank and fall together. If they
# fall at the same rank, "the rule is used exactly when the host cannot solve
# the task without it" is a measurement rather than a story.
#
# Nothing here touches the rule model or the adapter: it is the existing lora
# stem at different ranks, with RUN_SUFFIX keeping the runs apart.
#
# Usage:
#   ./scripts/run_plasticity_ablation.sh              # train, then evaluate
#   STAGES="eval" ./scripts/run_plasticity_ablation.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES="${STAGES:-train eval}"
SEEDS="${SEEDS:-1}"
RANKS="${RANKS:-1 4 16 64 256}"
BASE="${BASE:-ckpt/cp_bass_pre_bar_s1/cp_bass_pre_bar_s1.step004000.ckpt}"

# Same configuration as the per-bar experiment, so the new points are
# comparable with the frozen and trainar runs already in checkpoints_bar_fix.
export TRAIN_DATA="${TRAIN_DATA:-data/bass_finetune_bar.pt}"
export PRETRAIN_DATA="${PRETRAIN_DATA:-data/bass_pretrain_bar.pt}"
export VAL_DATA="${VAL_DATA:-data/bass_all_bar.pt}"
export UNSEEN_DATA="${UNSEEN_DATA:-data/bass_unseen_bar.pt}"
export CKPT_DIR="${CKPT_DIR:-checkpoints_plasticity}"
export LOG_DIR="${LOG_DIR:-logs_plasticity}"
export N_SKIP="${N_SKIP:-1}"
export KEY_SELECT="${KEY_SELECT:-first}"
export BEATS_PER_CHORD="${BEATS_PER_CHORD:-4}"
export TRAIN_LENGTH="${TRAIN_LENGTH:-32}"
export PROBE_PROMPT="${PROBE_PROMPT:-4}"
export PROBE_GEN="${PROBE_GEN:-28}"
export FREEZE_BASE_EVAL="${FREEZE_BASE_EVAL:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export BASE_CKPT="$BASE"

MAX_POLY="${MAX_POLY:-2}"
N_TRIALS="${N_TRIALS:-20}"
TEMPERATURE="${TEMPERATURE:-0}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
has () { [[ " $STAGES " == *" $1 "* ]]; }

if [[ ! -f "$BASE" ]]; then
  echo "MISSING base checkpoint: $BASE" >&2; exit 1
fi

# --- train one lora stem per rank -------------------------------------------
if has train; then
  for sd in $SEEDS; do
    for r in $RANKS; do
      echo "=============== seed $sd, lora rank $r ==============="
      SEED="$sd" RUN_SUFFIX="_r${r}_s${sd}" LORA_RANK="$r" \
        ./scripts/run_cp_stems.sh lora \
        >| "$LOG_DIR/launch_r${r}_s${sd}.log" 2>&1 \
        || echo "  rank $r seed $sd FAILED (see $LOG_DIR/launch_r${r}_s${sd}.log)" >&2
    done
  done
fi

# --- one table across the whole sweep ---------------------------------------
# The frozen and fully-trainable endpoints live in the main experiment's
# directory, so they are named explicitly rather than re-trained.
if has eval; then
  for sd in $SEEDS; do
    runs=()
    for r in $RANKS; do runs+=("cp_yy_lora_r${r}_s${sd}"); done

    echo
    echo "=============== table, seed $sd ==============="
    python -u -m midi_adapter.cp_results_table \
      --base_ckpt "$BASE" --ckpt_dir "$CKPT_DIR" --runs "${runs[@]}" \
      --adapter_rank 256 --n_skip "$N_SKIP" \
      --max_polyphony "$MAX_POLY" --beats_per_chord "$BEATS_PER_CHORD" \
      --n_prompt_beats "$PROBE_PROMPT" --n_gen_beats "$PROBE_GEN" \
      --n_trials "$N_TRIALS" --temperature "$TEMPERATURE" \
      --rule_key_select "$KEY_SELECT" \
      --json_out "$LOG_DIR/table_r_sweep_s${sd}.json" \
      2>&1 | tee "$LOG_DIR/table_r_sweep_s${sd}.log"

    # Rule dependence at each rank. acc_wrong near 0 means the channel is
    # load-bearing; near acc_true means the host is doing it alone.
    echo
    echo "=============== rule dependence, seed $sd ==============="
    for r in $RANKS; do
      run="cp_yy_lora_r${r}_s${sd}"
      ck=$(ls -1 "$CKPT_DIR/$run"/*unseen_acc*.ckpt 2>/dev/null \
           | sort -t= -k3 -g | tail -1)
      if [[ -z "$ck" ]]; then echo "  $run: no checkpoint"; continue; fi
      echo "--------------- $run"
      # --lora_rank must be passed: debug_frozen_adapter does not recover it
      # from the checkpoint the way cp_results_table does.
      python -u -m midi_adapter.debug_frozen_adapter \
        --base_ckpt "$BASE" --adapter_ckpt "$ck" --lora_rank "$r" \
        --n_skip "$N_SKIP" --beats_per_chord "$BEATS_PER_CHORD" \
        --rule_key_select "$KEY_SELECT" --max_polyphony "$MAX_POLY" \
        --n_prompt_beats "$PROBE_PROMPT" --n_gen_beats "$PROBE_GEN" \
        --n_trials "$N_TRIALS"
    done 2>&1 | tee "$LOG_DIR/sensitivity_sweep_s${sd}.log"
  done
fi

echo
echo "done."
echo "  tables      : $LOG_DIR/table_r_sweep_s<seed>.log"
echo "  sensitivity : $LOG_DIR/sensitivity_sweep_s<seed>.log"
echo "  endpoints   : frozen and trainar are in checkpoints_bar_fix, already measured"
true
