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
# Stages, any subset, in this order (the v3 pipeline's convention):
#   train   the stems for each seed
#   eval    the table (and MIDI) for each seed, then the combined table
#
# Useful:
#   TAG         names everything: checkpoints_$TAG, logs_$TAG, eval_midi_$TAG
#               and the run suffix _$TAG_s<seed>. Default v4.
#   BASE_CKPT   the pretrained host. Default is the v0.42 checkpoint.
#   SEEDS       default "1 2 3"
#   STEMS       default "finetune frozen lora noinput trainar scratch"
#   TEMPS       default "0 1"
#   PARALLEL=0  run the stems of a wave one after another instead of together
#
# Usage, in the shape of the v3 launch:
#   cd /l/users/xinyue.li/rasp && \
#   TAG=v4 STAGES="train eval" TEMPS="0" \
#     nohup ./scripts/run_v4_stems.sh >| bass_v4_driver.log 2>&1 &

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES="${STAGES:-train eval}"
# EVAL_ONLY=1 from the bar experiment still means "just the tables".
[[ "${EVAL_ONLY:-0}" == "1" ]] && STAGES="eval"
has () { [[ " $STAGES " == *" $1 "* ]]; }
TAG="${TAG:-v4}"
export BASE_CKPT="${BASE_CKPT:-checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt}"
SEEDS="${SEEDS:-1 2 3}"
STEMS="${STEMS:-finetune frozen lora noinput trainar scratch}"
PARALLEL="${PARALLEL:-1}"

# --- the v4 configuration --------------------------------------------------
export TRAIN_DATA="${TRAIN_DATA:-data/v4_bass_finetune.pt}"
# v4 has no pretrain-key set: the host pretrained on the whole LA corpus, so
# every key is "seen in pretrain" and the split is two-way, finetune keys vs
# the held-out F# and G#. The scratch stem still wants a second file to
# interleave; it gets the finetune file, so scratch = a random-init host
# trained with the adapter on the finetune keys.
export PRETRAIN_DATA="${PRETRAIN_DATA:-data/v4_bass_finetune.pt}"
export VAL_DATA="${VAL_DATA:-data/v4_bass_all.pt}"
export UNSEEN_DATA="${UNSEEN_DATA:-data/v4_bass_unseen.pt}"
export CKPT_DIR="${CKPT_DIR:-checkpoints_$TAG}"
export LOG_DIR="${LOG_DIR:-logs_$TAG}"
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
MIDI_DIR="${MIDI_DIR-eval_midi_$TAG}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# --- preflight -------------------------------------------------------------
missing=0
[[ -f "$BASE_CKPT" ]] || { echo "MISSING host: $BASE_CKPT" >&2; missing=1; }
for f in "$TRAIN_DATA" "$VAL_DATA" "$UNSEEN_DATA"; do
  [[ -f "$f" ]] || { echo "MISSING data: $f  (scp data/v4_bass_* from the Mac)" >&2; missing=1; }
done
[[ "$missing" == "1" ]] && exit 1
# The held-out keys must be unreachable: no window may be transposed (every
# pitch_shift_range must be [0,0]) and no finetune span may carry a held-out
# key. Checked here because FramedDataset applies the ranges silently.
python - "$TRAIN_DATA" "$UNSEEN_DATA" <<'PYCHECK' || exit 1
import sys, torch
tr, un = sys.argv[1], sys.argv[2]
held = set(int(k) for k in torch.load(un[:-3] + '.keys.pt').tolist())
ft = set(int(k) for k in torch.load(tr[:-3] + '.keys.pt').tolist())
bad = []
for f in (tr, un):
    psr = torch.load(f[:-3] + '.pitch_shift_range.pt')
    if int(psr.abs().sum()):
        bad.append(f'{f}: pitch_shift_range is not all zero -- windows would be transposed')
if ft & held:
    bad.append(f'{tr}: carries held-out keys {sorted(ft & held)}')
if bad:
    print('\n'.join('REFUSING: ' + b for b in bad), file=sys.stderr); sys.exit(1)
print(f'keys : finetune {sorted(ft)}  unseen {sorted(held)}  (no transposition)')
PYCHECK
echo "host : $BASE_CKPT"
echo "tag  : $TAG   stages: $STAGES   seeds: $SEEDS   stems: $STEMS"
echo "data : $TRAIN_DATA / $UNSEEN_DATA / $PRETRAIN_DATA / $VAL_DATA"

# --- the table for one seed ------------------------------------------------
eval_seed () {
  local sd="$1" ft args t tag
  ft="$(ls -1 "$CKPT_DIR/cp_bass_ft_${TAG}_s$sd"/*.ckpt 2>/dev/null \
        | grep -v 'last\.ckpt' | sort -t= -k3 -g | head -1)"
  if [[ -n "$ft" ]]; then echo "  finetune baseline: $(basename "$ft")"
  else echo "  finetune baseline: none found -- the Finetune column is omitted"; fi

  for t in $TEMPS; do
    tag="p${PROBE_PROMPT}_T${t}"
    args=(--base_ckpt "$BASE_CKPT" --ckpt_dir "$CKPT_DIR"
          --runs "cp_yy_frozen_${TAG}_s$sd" "cp_yy_lora_${TAG}_s$sd" "cp_yy_trainar_${TAG}_s$sd"
                 "cp_yy_scratch_${TAG}_s$sd" "cp_yy_noinput_mlp_aux1_${TAG}_s$sd"
          --adapter_rank 256 --n_skip "$N_SKIP" --rule_key_select "$KEY_SELECT"
          --prompt_data "$TRAIN_DATA" "$UNSEEN_DATA"
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
  export SEED="$sd" RUN_SUFFIX="_${TAG}_s$sd"
  echo "=== seed $sd ==="
  if [[ "$PARALLEL" != "1" ]]; then
    for s in $STEMS; do
      ./scripts/run_cp_stems.sh "$s" >| "$LOG_DIR/launch_${s}_${TAG}_s${sd}.log" 2>&1 \
        && echo "  $s: done" || echo "  $s: FAILED (see $LOG_DIR/launch_${s}_${TAG}_s${sd}.log)"
    done
    return
  fi
  for s in $STEMS; do
    ./scripts/run_cp_stems.sh "$s" >| "$LOG_DIR/launch_${s}_${TAG}_s${sd}.log" 2>&1 &
    pids+=($!)
    echo "  $s pid=${pids[-1]}"
    sleep 45
  done
  local i=0
  for s in $STEMS; do
    if wait "${pids[$i]}"; then echo "  $s: done"
    else echo "  $s: FAILED (see $LOG_DIR/launch_${s}_${TAG}_s${sd}.log)"; fi
    i=$((i + 1))
  done
}

for sd in $SEEDS; do
  has train && train_seed "$sd"
  has eval && eval_seed "$sd"
done

has eval || { echo; echo "train stage done (no eval requested)."; exit 0; }
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
