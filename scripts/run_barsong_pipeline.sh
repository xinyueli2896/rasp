#!/usr/bin/env bash
# The per-SONG realisation pipeline: data, one pretrain, then the three-seed
# experiment.
#
# The existing bar datasets were generated before commit 1d06d93 ("One bar
# pattern per song, not per bar"), so they draw a realisation independently
# per bar -- 0 of 3000 songs keep one. That makes pattern-following
# unmeasurable: there is no per-song latent to carry, and every model sitting
# at chance was correctly reproducing its data. These splits have the latent,
# so the question becomes askable.
#
# Only ONE pretrain is needed: the experiment pins the host across seeds with
# BASE_CKPT_SHARED, so seeds vary the adapter init and the batch order only.
#
# Stages, any subset, in this order:
#   data      generate and verify the four splits
#   pretrain  one host on the pretrain keys
#   train     three seeds x six models, tables, MIDI
#
# Usage:
#   ./scripts/run_barsong_pipeline.sh                 # all three
#   STAGES="data" ./scripts/run_barsong_pipeline.sh   # just the data

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGES="${STAGES:-data pretrain train}"
TAG="${TAG:-barsong}"
N_SONGS="${N_SONGS:-3000}"
N_UNSEEN_SONGS="${N_UNSEEN_SONGS:-1000}"   # 2 keys x 500, matching the old set
N_BARS="${N_BARS:-32}"
MAX_POLY="${MAX_POLY:-2}"
PRE_STEP="${PRE_STEP:-004000}"
PRE_NAME="cp_bass_pre_${TAG}"

# The bar-cycle geometry. A cycle is 16 beats, so windows must start on a
# multiple of 16 or the rule model's phase is wrong for the whole window.
TRAIN_LENGTH="${TRAIN_LENGTH:-32}"
SAMPLE_STEP="${SAMPLE_STEP:-16}"
MAX_STEPS="${MAX_STEPS:-10000}"

has () { [[ " $STAGES " == *" $1 "* ]]; }

# --- data -------------------------------------------------------------------
if has data; then
  echo "=============== generating $TAG splits ==============="
  gen () {   # gen <name> <n_songs> <keys...>
    local name="$1" n="$2"; shift 2
    echo "--- $name : keys $* ---"
    python -m midi_adapter.generate_synthetic_bass \
      --n_songs "$n" --n_bars "$N_BARS" --max_polyphony "$MAX_POLY" \
      --cycle_unit bar --no_midi --keys "$@" \
      --out_dir "data/bass_${name}_${TAG}" --out_pt "data/bass_${name}_${TAG}" \
      || { echo "generation failed for $name" >&2; exit 1; }
  }
  # Key split, unchanged: pretrain {0,5,9}, finetune {0,1,2,3,4,7,10,11},
  # unseen {6,8}. Key 0 belongs in BOTH pretrain and finetune -- leaving it out
  # of the finetune split is what made "Both seen" meaningless last time.
  gen pretrain "$N_SONGS"        0 5 9
  gen finetune "$N_SONGS"        0 1 2 3 4 7 10 11
  gen unseen   "$N_UNSEEN_SONGS" 6 8
  gen all      "$N_SONGS"        0 1 2 3 4 5 6 7 8 9 10 11

  echo
  echo "=============== verifying ==============="
  python -m midi_adapter.verify_bass_data \
      "data/bass_pretrain_${TAG}" "data/bass_finetune_${TAG}" \
      "data/bass_unseen_${TAG}" "data/bass_all_${TAG}" \
      2>&1 | tee "verify_${TAG}.log"

  # Hard gate: the whole point of this run is the per-song latent. If any song
  # mixes realisations the data is the old kind and everything downstream is
  # measuring nothing.
  if grep -q 'mix realisations' "verify_${TAG}.log"; then
    echo >&2
    echo "ABORT: songs mix realisations -- this is the pre-1d06d93 behaviour." >&2
    echo "The generator should draw one pattern per song. Nothing downstream" >&2
    echo "is worth running until that holds." >&2
    exit 1
  fi
  echo "data ok: one realisation per song"
fi

# --- pretrain ---------------------------------------------------------------
if has pretrain; then
  echo
  echo "=============== pretrain (one host for all seeds) ==============="
  python -m midi_adapter.pretrain_cp_bass \
    --train_data "data/bass_pretrain_${TAG}.pt" \
    --val_data   "data/bass_all_${TAG}.pt" \
    --model_size 1 --batch_size 8 \
    --train_length "$TRAIN_LENGTH" --sample_step "$SAMPLE_STEP" \
    --max_steps "$MAX_STEPS" --val_check_interval 500 --save_every_n_steps 500 \
    --seed 42 --run_name "$PRE_NAME" --ckpt_dir ckpt \
    --no_tensorboard --wandb_project cp_bass \
    2>&1 | tee "pretrain_${TAG}.log"
fi

BASE="ckpt/${PRE_NAME}/${PRE_NAME}.step${PRE_STEP}.ckpt"

# --- train ------------------------------------------------------------------
if has train; then
  if [[ ! -f "$BASE" ]]; then
    echo "MISSING pretrain checkpoint: $BASE" >&2
    echo "Pick one with: ls -1 ckpt/${PRE_NAME}/ ; set PRE_STEP=<nnnnnn>" >&2
    exit 1
  fi
  echo
  echo "=============== three seeds from $BASE ==============="
  BASE_CKPT_SHARED="$BASE" \
  TRAIN_DATA="data/bass_finetune_${TAG}.pt" \
  PRETRAIN_DATA="data/bass_pretrain_${TAG}.pt" \
  VAL_DATA="data/bass_all_${TAG}.pt" \
  UNSEEN_DATA="data/bass_unseen_${TAG}.pt" \
  CKPT_DIR="checkpoints_${TAG}" LOG_DIR="logs_${TAG}" \
  MIDI_DIR="eval_midi_${TAG}" \
  ./scripts/run_bar_experiment.sh
fi

echo
echo "done. next:"
echo "  tables   : logs_${TAG}/table_all_seeds_T0.log  (and _T1)"
echo "  patterns : python -m midi_adapter.prompt_pattern_sweep --base_ckpt $BASE \\"
echo "               --ckpt_dir checkpoints_${TAG} --runs cp_yy_frozen_s1 ..."
true
