#!/usr/bin/env bash
# The v6 experiment: five adapter stems plus the no-adapter baseline, trained
# against a CHORD cadence instead of a root cadence.
#
#   ./scripts/run_v6_stems.sh                    # train, then the table
#   STAGES=eval ./scripts/run_v6_stems.sh        # table only
#   SEEDS=1 STEMS=frozen ./scripts/run_v6_stems.sh
#
# What differs from run_v4_stems.sh, and nothing else does:
#
#   RULE_MODE=chord   the adapter reads ChordTracrRuleModel's next-chord block
#                     (24 of its 52 dims) instead of the compiled root model's
#                     root_next. The symbol IS the chord, so there is no
#                     separate quality channel.
#   DATA_PREFIX       the v6 datasets, whose split is on tonic CHORDS -- F#,
#                     D#m, G# and Fm held out, 20 of the 24 trained
#   the table         v6_eval, which scores an exact chord match and names
#                     which cadence the generation followed
#
# Everything else is v4/v5's configuration unchanged, deliberately, so the
# tables compare: BEATS_PER_CHORD=16, TRAIN_LENGTH=128, a 16-subbeat prompt
# with 112 generated, KEY_SELECT=first and RULE_RATE=bar.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES="${STAGES:-train eval}"
[[ "${EVAL_ONLY:-0}" == "1" ]] && STAGES="eval"
has () { [[ " $STAGES " == *" $1 "* ]]; }

TAG="${TAG:-v6}"
WORK="${WORK:-$REPO/v6}"
SEEDS="${SEEDS:-1 2 3}"
ALL_STEMS="finetune frozen lora noinput trainar scratch"
STEMS="${STEMS:-$ALL_STEMS}"
EVAL_STEMS="${EVAL_STEMS:-$STEMS}"
PARALLEL="${PARALLEL:-1}"

export BASE_CKPT="${BASE_CKPT:-$WORK/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"

run_of () {
  case "$1" in
    finetune) echo cp_bass_ft ;;        frozen)  echo cp_yy_frozen ;;
    lora)     echo cp_yy_lora ;;        trainar) echo cp_yy_trainar ;;
    scratch)  echo cp_yy_scratch ;;     noinput) echo cp_yy_noinput_mlp_aux1 ;;
    *) echo "unknown stem: $1" >&2; exit 1 ;;
  esac
}

# --- the v6 configuration --------------------------------------------------
DATA_PREFIX="${DATA_PREFIX:-$WORK/data/v6_chordal}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_PREFIX}_finetune.pt}"
# The host was pretrained on the whole LA corpus, so there is no pretrain-key
# set and the split is two-way: the 20 trained tonic chords against the four
# held out. scratch still needs a second file to interleave; it gets the
# finetune file, making it a random-init host trained with the adapter.
export PRETRAIN_DATA="${PRETRAIN_DATA:-${DATA_PREFIX}_finetune.pt}"
export VAL_DATA="${VAL_DATA:-${DATA_PREFIX}_all.pt}"
export UNSEEN_DATA="${UNSEEN_DATA:-${DATA_PREFIX}_unseen.pt}"
export CKPT_DIR="${CKPT_DIR:-$WORK/checkpoints_$TAG}"
export LOG_DIR="${LOG_DIR:-$WORK/logs_$TAG}"

# v6's rule. run_cp_stems.sh passes these through to the trainer.
export RULE_MODE="${RULE_MODE:-chord}"
export RULE_VIEW="${RULE_VIEW:-root_next}"   # -> chord_next for the chord model
export CHORD_MINOR="${CHORD_MINOR:-harmonic}"
export WANDB_PROJECT="${WANDB_PROJECT:-cp_bass}"

# Unchanged from v4/v5 so the tables are comparable.
export N_SKIP="${N_SKIP:-1}"
export KEY_SELECT="${KEY_SELECT:-first}"
export RULE_RATE="${RULE_RATE:-bar}"
export STRICT_CAUSAL="${STRICT_CAUSAL:-0}"
export LORA_RANK="${LORA_RANK:-16}"
export GATE_INIT="${GATE_INIT:-0.0}"
export FREEZE_BASE_EVAL="${FREEZE_BASE_EVAL:-1}"
export BEATS_PER_CHORD="${BEATS_PER_CHORD:-16}"
export TRAIN_LENGTH="${TRAIN_LENGTH:-128}"
export PROBE_PROMPT="${PROBE_PROMPT:-16}"
export PROBE_GEN="${PROBE_GEN:-112}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

TEMPS="${TEMPS:-0}"
N_TRIALS="${N_TRIALS:-5}"
MIDI_DIR="${MIDI_DIR-$WORK/eval_midi_$TAG}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# --- preflight -------------------------------------------------------------
for f in "$BASE_CKPT" "$TRAIN_DATA" "$UNSEEN_DATA"; do
  [[ -s "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
echo "host    $BASE_CKPT"
echo "data    $DATA_PREFIX  (rule $RULE_MODE/$RULE_VIEW, $CHORD_MINOR minor)"
echo "seeds   $SEEDS"
echo "stems   $STEMS"

# Each stem's own output goes to its launch log, so a failure is only visible
# here as an exit status -- which has to be checked and propagated. An earlier
# version printed "done" unconditionally in the serial path, and a stem that
# died in three seconds on an ImportError reported success; inside a job array
# that is 18 tasks all claiming to have trained.
train_seed () {
  local sd="$1" s pids=() failed=0 log
  export SEED="$sd" RUN_SUFFIX="_${TAG}_s$sd"
  echo "=== seed $sd ==="
  for s in $STEMS; do
    log="$LOG_DIR/launch_${s}_${TAG}_s${sd}.log"
    if [[ "$PARALLEL" == 1 ]]; then
      ./scripts/run_cp_stems.sh "$s" >| "$log" 2>&1 &
      pids+=($!); echo "  $s pid=${pids[-1]}"; sleep 30
    elif ./scripts/run_cp_stems.sh "$s" >| "$log" 2>&1; then
      echo "  $s: done"
    else
      echo "  $s: FAILED (exit $?) -- $log" >&2
      tail -5 "$log" | sed 's/^/      /' >&2
      failed=1
    fi
  done
  local i=0
  for s in $STEMS; do
    if [[ "$PARALLEL" == 1 ]]; then
      log="$LOG_DIR/launch_${s}_${TAG}_s${sd}.log"
      if wait "${pids[$i]}"; then echo "  $s: done"
      else
        echo "  $s: FAILED -- $log" >&2
        tail -5 "$log" | sed 's/^/      /' >&2
        failed=1
      fi
      i=$((i + 1))
    fi
  done
  return "$failed"
}

eval_seed () {
  local sd="$1" t args runs=()
  for s in $EVAL_STEMS; do
    runs+=("$(run_of "$s")_${TAG}_s$sd")
  done
  for t in $TEMPS; do
    args=(--data_prefix "$DATA_PREFIX" --base_ckpt "$BASE_CKPT"
          --ckpt_dir "$CKPT_DIR" --runs "${runs[@]}"
          --adapter_rank 256 --n_skip "$N_SKIP"
          --beats_per_chord "$BEATS_PER_CHORD"
          --n_prompt_beats "$PROBE_PROMPT" --n_gen_beats "$PROBE_GEN"
          --n_trials "$N_TRIALS" --temperature "$t"
          --minor_pattern "$CHORD_MINOR" --rule_view "$RULE_VIEW"
          --json_out "$LOG_DIR/table_s${sd}_T${t}.json")
    [[ -n "$MIDI_DIR" ]] && args+=(--save_midi_dir "$MIDI_DIR/s$sd/T$t")
    echo "=== table, seed $sd, T=$t ==="
    python -m midi_adapter.v6_eval "${args[@]}" \
      2>&1 | tee "$LOG_DIR/table_s${sd}_T${t}.log"
  done
}

rc=0
for sd in $SEEDS; do
  if has train; then train_seed "$sd" || rc=1; fi
  has eval && eval_seed "$sd"
done

echo
if (( rc )); then
  echo "FINISHED WITH FAILURES -- see the launch logs named above." >&2
else
  echo "done."
for sd in $SEEDS; do for t in $TEMPS; do
  echo "  table  $LOG_DIR/table_s${sd}_T${t}.log"
done; done
[[ -n "$MIDI_DIR" ]] && echo "  midi   $MIDI_DIR/s<seed>/T<temp>/<model>/"
fi
# Non-zero so a SLURM array task that trained nothing is marked FAILED rather
# than passing quietly.
exit "$rc"
