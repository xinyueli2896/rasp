#!/usr/bin/env bash
# The latent-rule (one chord per bar) bass experiment, end to end:
# three seeds x six models, then the accuracy table for each seed, with the
# table starting automatically as soon as that seed's training finishes.
#
# Six models per seed: the no-adapter finetune baseline plus the five adapter
# stems. Seeds run as waves rather than all at once -- three of the six train
# the full 112M host (~3.5 GB each) against ~1.4 GB for the frozen-host stems,
# so one wave is ~15 GB and all eighteen would be ~45 GB.
#
# Evaluating after each wave rather than at the very end means seed 1's table
# is readable while seeds 2 and 3 are still training, so a misconfiguration
# shows up after a third of the compute instead of all of it.
#
# Required:
#   BASE_FMT    printf format for the base checkpoint, with %s for the seed.
#               Default assumes the step004000 snapshot of cp_bass_pre_bar_s<seed>.
# Useful:
#   SEEDS       default "1 2 3"
#   EVAL_ONLY=1 skip training, just rebuild the tables from existing checkpoints
#   STEMS       default "finetune frozen lora noinput trainar scratch"
#
# Usage:  ./scripts/run_bar_experiment.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SEEDS="${SEEDS:-1 2 3}"
STEMS="${STEMS:-finetune frozen lora noinput trainar scratch}"
EVAL_ONLY="${EVAL_ONLY:-0}"
BASE_FMT="${BASE_FMT:-ckpt/cp_bass_pre_bar_s%s/cp_bass_pre_bar_s%s.step004000.ckpt}"

# --- the latent-rule configuration -----------------------------------------
export TRAIN_DATA="${TRAIN_DATA:-data/bass_finetune_bar.pt}"
export PRETRAIN_DATA="${PRETRAIN_DATA:-data/bass_pretrain_bar.pt}"
export VAL_DATA="${VAL_DATA:-data/bass_all_bar.pt}"
export UNSEEN_DATA="${UNSEEN_DATA:-data/bass_unseen_bar.pt}"
export CKPT_DIR="${CKPT_DIR:-checkpoints_bar}"
export LOG_DIR="${LOG_DIR:-logs_bar}"
export N_SKIP="${N_SKIP:-1}"
export KEY_SELECT="${KEY_SELECT:-first}"     # required: phase0 would average a whole bar
export STRICT_CAUSAL="${STRICT_CAUSAL:-0}"
export LORA_RANK="${LORA_RANK:-16}"
export GATE_INIT="${GATE_INIT:-0.0}"
export FREEZE_BASE_EVAL="${FREEZE_BASE_EVAL:-1}"
export BEATS_PER_CHORD="${BEATS_PER_CHORD:-4}"
export TRAIN_LENGTH="${TRAIN_LENGTH:-32}"
export PROBE_PROMPT="${PROBE_PROMPT:-4}"
export PROBE_GEN="${PROBE_GEN:-28}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Evaluation protocol. These must match the data or the table is wrong in
# silence: max_polyphony sets the prompt's token width, beats_per_chord makes
# it score downbeats rather than every beat.
MAX_POLY="${MAX_POLY:-2}"
N_TRIALS="${N_TRIALS:-20}"
# One MIDI per key per adapter run, generated with the same prompts the table
# scores, so what you hear matches the numbers. Set to "" to skip.
MIDI_DIR="${MIDI_DIR-eval_midi_bar}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

base_for () { printf "$BASE_FMT" "$1" "$1"; }

# --- preflight: every base must exist before anything is launched ----------
missing=0
for sd in $SEEDS; do
  b="$(base_for "$sd")"
  if [[ ! -f "$b" ]]; then echo "MISSING base for seed $sd: $b" >&2; missing=1; fi
done
if [[ "$missing" == "1" ]]; then
  echo "Set BASE_FMT, or wait for the pretrains to reach that step." >&2
  exit 1
fi

# --- the table for one seed ------------------------------------------------
eval_seed () {
  local sd="$1" base ft args
  base="$(base_for "$sd")"

  # The finetune baseline is selected on val_loss, which is what its filenames
  # carry. Absent (or still training) -> the column is simply omitted.
  ft="$(ls -1 "$CKPT_DIR/cp_bass_ft_s$sd"/*.ckpt 2>/dev/null \
        | grep -v 'last\.ckpt' | sort -t= -k3 -g | head -1)"

  args=(--base_ckpt "$base" --ckpt_dir "$CKPT_DIR"
        --runs "cp_yy_frozen_s$sd" "cp_yy_lora_s$sd" "cp_yy_trainar_s$sd"
               "cp_yy_scratch_s$sd" "cp_yy_noinput_mlp_aux1_s$sd"
        --adapter_rank 256 --n_skip "$N_SKIP"
        --max_polyphony "$MAX_POLY" --beats_per_chord "$BEATS_PER_CHORD"
        --n_prompt_beats "$PROBE_PROMPT" --n_gen_beats "$PROBE_GEN"
        --n_trials "$N_TRIALS"
        --rule_key_select "$KEY_SELECT")
  if [[ -n "$MIDI_DIR" ]]; then
    args+=(--save_midi_dir "$MIDI_DIR/s$sd")
  fi
  args+=(--json_out "$LOG_DIR/table_s$sd.json")
  if [[ -n "$ft" ]]; then
    args+=(--ft_ckpt "$ft")
    echo "  finetune baseline: $(basename "$ft")"
  else
    echo "  finetune baseline: none found -- the Finetune column is omitted"
  fi

  echo "=== table, seed $sd ==="
  python -m midi_adapter.cp_results_table "${args[@]}" \
      2>&1 | tee "$LOG_DIR/table_s$sd.log"
}

# --- one wave: six models for one seed, in parallel ------------------------
train_seed () {
  local sd="$1" s pids=()
  export SEED="$sd" RUN_SUFFIX="_s$sd"
  export BASE_CKPT="$(base_for "$sd")"
  echo "=== seed $sd : $BASE_CKPT ==="
  for s in $STEMS; do
    ./scripts/run_cp_stems.sh "$s" >| "$LOG_DIR/launch_${s}_s${sd}.log" 2>&1 &
    pids+=($!)
    echo "  $s pid=${pids[-1]}"
    sleep 45
  done
  # Wait on each explicitly and report, so one stem dying does not take the
  # wave down with it -- the table still runs on whatever finished.
  local i=0
  for s in $STEMS; do
    if wait "${pids[$i]}"; then echo "  $s: done"
    else echo "  $s: FAILED (see $LOG_DIR/launch_${s}_s${sd}.log)"; fi
    i=$((i + 1))
  done
}

for sd in $SEEDS; do
  if [[ "$EVAL_ONLY" != "1" ]]; then train_seed "$sd"; fi
  eval_seed "$sd"
done

# --- one combined table across seeds -------------------------------------
jsons=""
for sd in $SEEDS; do
  [[ -f "$LOG_DIR/table_s$sd.json" ]] && jsons="$jsons $LOG_DIR/table_s$sd.json"
done
if [[ -n "$jsons" ]]; then
  echo
  python -m midi_adapter.aggregate_seeds $jsons \
      2>&1 | tee "$LOG_DIR/table_all_seeds.log"
fi

echo
echo "all seeds done."
for sd in $SEEDS; do echo "  per seed : $LOG_DIR/table_s$sd.log"; done
echo "  combined : $LOG_DIR/table_all_seeds.log"
[[ -n "$MIDI_DIR" ]] && echo "  midi     : $MIDI_DIR/s<seed>/<model>/"
true
