#!/usr/bin/env bash
# v4: the six models on REAL bass lines, with the pretrained LA host.
#
# Same shape as run_bar_experiment.sh -- one wave of six models per seed,
# then the table -- with three things changed, all of them data:
#
#   * The host is the pretrained cp_transformer_v0.42_size1 checkpoint, the
#     same for every seed, so the seed varies only the adapter init and the
#     batch order. There is no per-seed pretrain: the host IS the pretrain.
#   * The data is data/v4_bass_*.pt from prepare_v4_data: bass lines from the
#     LA MIDI corpus that already follow I-IV-V-I, 16 subbeats to the bar at
#     polyphony 16. One chord spans 16 steps, a cycle 64, so the 2-cycle
#     window of v2/v3 (32 beats) is 128 steps here, and the 1-bar prompt is 16.
#   * The table is v4_eval, which prompts with real spans from the val split
#     instead of the synthetic generator, and reports the Data row.
#
# The rule model and the adapter are untouched: KEY_SELECT=first and
# RULE_RATE=bar are the v2_align configuration, just read through
# BEATS_PER_CHORD=16.
#
# Required:
#   BASE_CKPT   the pretrained host, e.g.
#               checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
# Useful:
#   SEEDS       default "1 2 3"
#   STEMS       default "finetune frozen lora noinput trainar scratch"
#   EVAL_ONLY=1 skip training, rebuild the tables from existing checkpoints
#   PARALLEL=0  run the stems of a wave one after another instead of together
#
# Usage:  BASE_CKPT=checkpoints/....ckpt ./scripts/run_v4_stems.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${BASE_CKPT:?set BASE_CKPT to the pretrained v0.42 host checkpoint}"
export BASE_CKPT
SEEDS="${SEEDS:-1 2 3}"
STEMS="${STEMS:-finetune frozen lora noinput trainar scratch}"
EVAL_ONLY="${EVAL_ONLY:-0}"
PARALLEL="${PARALLEL:-1}"

# --- the v4 configuration --------------------------------------------------
export TRAIN_DATA="${TRAIN_DATA:-data/v4_bass_finetune.pt}"
export PRETRAIN_DATA="${PRETRAIN_DATA:-data/v4_bass_pretrain.pt}"
export VAL_DATA="${VAL_DATA:-data/v4_bass_all.pt}"
export UNSEEN_DATA="${UNSEEN_DATA:-data/v4_bass_unseen.pt}"
export CKPT_DIR="${CKPT_DIR:-checkpoints_v4}"
export LOG_DIR="${LOG_DIR:-logs_v4}"
export N_SKIP="${N_SKIP:-1}"
export KEY_SELECT="${KEY_SELECT:-first}"   # the slot holding the downbeat's root
export RULE_RATE="${RULE_RATE:-bar}"       # one key per bar: v2_align
export STRICT_CAUSAL="${STRICT_CAUSAL:-0}"
export LORA_RANK="${LORA_RANK:-16}"
export GATE_INIT="${GATE_INIT:-0.0}"
export FREEZE_BASE_EVAL="${FREEZE_BASE_EVAL:-1}"
# 16 subbeats per bar (beat_div 4): the root is held 16 steps, the cycle is
# 64, and the window is two cycles like v2/v3's 32 beats. Window starts are
# forced onto cycle boundaries by run_cp_stems.sh (sample_step = 4 * 16).
export BEATS_PER_CHORD="${BEATS_PER_CHORD:-16}"
export TRAIN_LENGTH="${TRAIN_LENGTH:-128}"
# In-training probe: one real bar as prompt, seven scored downbeats after it.
# Same protocol as the table below, so checkpoint selection and the reported
# number agree.
export PROBE_PROMPT="${PROBE_PROMPT:-16}"
export PROBE_GEN="${PROBE_GEN:-112}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Evaluation protocol.
N_TRIALS="${N_TRIALS:-10}"
TEMPS="${TEMPS:-0 1}"
MIDI_DIR="${MIDI_DIR-eval_midi_v4}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# --- preflight -------------------------------------------------------------
missing=0
[[ -f "$BASE_CKPT" ]] || { echo "MISSING host: $BASE_CKPT" >&2; missing=1; }
for f in "$TRAIN_DATA" "$VAL_DATA" "$UNSEEN_DATA" "$PRETRAIN_DATA"; do
  [[ -f "$f" ]] || { echo "MISSING data: $f  (scp data/v4_bass_* from the Mac)" >&2; missing=1; }
done
[[ "$missing" == "1" ]] && exit 1
echo "host : $BASE_CKPT"
echo "data : $TRAIN_DATA / $UNSEEN_DATA / $PRETRAIN_DATA / $VAL_DATA"

# --- the table for one seed ------------------------------------------------
eval_seed () {
  local sd="$1" ft args t tag
  ft="$(ls -1 "$CKPT_DIR/cp_bass_ft_v4_s$sd"/*.ckpt 2>/dev/null \
        | grep -v 'last\.ckpt' | sort -t= -k3 -g | head -1)"
  if [[ -n "$ft" ]]; then echo "  finetune baseline: $(basename "$ft")"
  else echo "  finetune baseline: none found -- the Finetune column is omitted"; fi

  for t in $TEMPS; do
    tag="p${PROBE_PROMPT}_T${t}"
    args=(--base_ckpt "$BASE_CKPT" --ckpt_dir "$CKPT_DIR"
          --runs "cp_yy_frozen_v4_s$sd" "cp_yy_lora_v4_s$sd" "cp_yy_trainar_v4_s$sd"
                 "cp_yy_scratch_v4_s$sd" "cp_yy_noinput_mlp_aux1_v4_s$sd"
          --adapter_rank 256 --n_skip "$N_SKIP" --rule_key_select "$KEY_SELECT"
          --prompt_data "$TRAIN_DATA" "$UNSEEN_DATA" "$PRETRAIN_DATA"
          --beats_per_chord "$BEATS_PER_CHORD"
          --n_prompt_beats "$PROBE_PROMPT" --n_gen_beats "$PROBE_GEN"
          --n_trials "$N_TRIALS" --temperature "$t"
          --json_out "$LOG_DIR/table_s${sd}_${tag}.json")
    [[ -n "$MIDI_DIR" ]] && args+=(--save_midi_dir "$MIDI_DIR/s$sd/$tag")
    [[ -n "$ft" ]] && args+=(--ft_ckpt "$ft")
    echo "=== table, seed $sd, temperature $t ==="
    python -m midi_adapter.v4_eval "${args[@]}" 2>&1 | tee "$LOG_DIR/table_s${sd}_${tag}.log"
  done
}

# --- one wave: the stems for one seed --------------------------------------
train_seed () {
  local sd="$1" s pids=()
  export SEED="$sd" RUN_SUFFIX="_v4_s$sd"
  echo "=== seed $sd ==="
  if [[ "$PARALLEL" != "1" ]]; then
    for s in $STEMS; do
      ./scripts/run_cp_stems.sh "$s" >| "$LOG_DIR/launch_${s}_v4_s${sd}.log" 2>&1 \
        && echo "  $s: done" || echo "  $s: FAILED (see $LOG_DIR/launch_${s}_v4_s${sd}.log)"
    done
    return
  fi
  for s in $STEMS; do
    ./scripts/run_cp_stems.sh "$s" >| "$LOG_DIR/launch_${s}_v4_s${sd}.log" 2>&1 &
    pids+=($!)
    echo "  $s pid=${pids[-1]}"
    sleep 45
  done
  local i=0
  for s in $STEMS; do
    if wait "${pids[$i]}"; then echo "  $s: done"
    else echo "  $s: FAILED (see $LOG_DIR/launch_${s}_v4_s${sd}.log)"; fi
    i=$((i + 1))
  done
}

for sd in $SEEDS; do
  [[ "$EVAL_ONLY" != "1" ]] && train_seed "$sd"
  eval_seed "$sd"
done

for t in $TEMPS; do
  jsons=""
  for sd in $SEEDS; do
    f="$LOG_DIR/table_s${sd}_p${PROBE_PROMPT}_T${t}.json"; [[ -f "$f" ]] && jsons="$jsons $f"
  done
  if [[ -n "$jsons" ]]; then
    echo; echo "=== combined across seeds, ${PROBE_PROMPT}-step prompt, temperature $t ==="
    python -m midi_adapter.aggregate_seeds $jsons 2>&1 \
      | tee "$LOG_DIR/table_all_seeds_p${PROBE_PROMPT}_T${t}.log"
  fi
done
echo; echo "all seeds done."
true
