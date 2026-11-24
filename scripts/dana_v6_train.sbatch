#!/bin/bash
# Train the v6 stems on DANA's GPU partition, one array task per (seed, stem).
#
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env sbatch \
#       -o $PWD/v6/logs_v6/train-%A_%a.out scripts/dana_v6_train.sbatch
#
# ONE job, training every requested stem in sequence on its single GPU. Not a
# job array, because cscc-gpu-qos caps SUBMITTED jobs (queued plus running) at
# 2 per user and an array counts every task against that -- `--array=0-5%2`
# is six submitted jobs and is refused outright with
# QOSMaxSubmitJobPerUserLimit. The %2 throttle limits only what RUNS.
#
# So submit one seed at a time: 6 stems sequentially is roughly 2-3 hours
# (frozen trains at ~9.5 it/s, so 10k steps is ~18 min; trainar and scratch
# train the full 112M host and take several times that), well inside the 24 h
# limit, and one submitted job leaves the other slot free for the eval.
#
#   WORK=... SEEDS=1 sbatch scripts/dana_v6_train.sbatch
#   WORK=... SEEDS=2 sbatch scripts/dana_v6_train.sbatch     # after, or alongside
#
# On a partition whose QoS allows more jobs -- `long` with gpu-12 permits 8 --
# an array is faster, and passing --array switches this script to one task per
# (seed, stem):
#   sbatch --partition=long --qos=gpu-12 --array=0-17%6 --time=48:00:00 ...
#
# An array rather than one long job because the stems are independent: a stem
# that dies costs its own task, not the run, and the queue can start them as
# GPUs free up instead of waiting for six at once.
#
#   SEEDS="1"              one seed (edit --array to match: stems - 1)
#   STEMS="frozen lora"    fewer stems (likewise)
#   MAX_STEPS=2000         a short smoke run before committing the full one
#   SHIFT_AUG=1            v0.42's pitch-shift augmentation, never into a
#                          held-out key (midi_adapter/shift_aug.py); use a new
#                          TAG so the runs do not overwrite the unaugmented ones
#
# PARTITION AND QOS. No QoS is requested in the directives, so the partition's
# own QoS applies: cscc-gpu-p carries cscc-gpu-qos. Pass one explicitly when
# you want the other: --qos=gpu-debug-qos for a smoke run, which is what DANA
# prescribes for environment and dependency checks.
#
# The eval is a separate submission once training is in: it needs every stem's
# checkpoint, so it cannot be an array task. It is PRODUCTION work -- it
# generates the results table -- so it takes the partition's production QoS by
# default and must NOT be given gpu-debug-qos, which DANA reserves for
# environment checks, dependency validation and short debugging runs.
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env STAGES=eval sbatch \
#       -o $PWD/v6/logs_v6/eval-%j.out scripts/dana_v6_train.sbatch
#
# Housekeeping, per DANA's data-management rule: once both stages exist the
# corpus zip under $WORK/LAMD is rebuildable and should go, as should the
# _min8 datasets if the production floor is what you train on. The stages
# themselves are kilobytes and worth keeping -- every dataset variation is a
# --from_stage rebuild off them.

#SBATCH --job-name=v6train
#SBATCH --output=v6train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cscc-gpu-p
#SBATCH --time=24:00:00

set -euo pipefail

WORK="${WORK:?set WORK to the v6 work directory, e.g. \$PWD/v6}"
CONDA_ENV="${CONDA_ENV:-$WORK/env}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SEEDS_ALL="${SEEDS:-1 2 3}"
STEMS_ALL="${STEMS:-finetune frozen lora noinput trainar scratch}"
STAGES="${STAGES:-train}"
# The experiment driver. run_v6_stems.sh by default; DRIVER=./scripts/run_v4_stems.sh
# runs v4/v5 through the same job (give it DATA_PREFIX, BASE_CKPT, CKPT_DIR and
# LOG_DIR under $WORK, since its own defaults are relative to the repo).
DRIVER="${DRIVER:-./scripts/run_v6_stems.sh}"

PY="${PY:-$CONDA_ENV/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "no interpreter at $PY." >&2
  echo "CONDA_ENV should be the env PREFIX (a path), e.g. \$PWD/v6/env." >&2
  exit 1
fi
if [[ ! -f "$REPO/midi_adapter/train_cp_yinyang.py" ]]; then
  echo "REPO=$REPO is not a checkout of this repository." >&2
  exit 1
fi
# Fail here, with something actionable, rather than three seconds into the
# first stem with an ImportError inside a per-stem launch log.
if ! "$PY" -c "import torch, transformers, pytorch_lightning, peft, wandb" 2>/dev/null; then
  echo "$PY is missing the training stack (transformers, pytorch_lightning, wandb," >&2
  echo "peft). Install it into the same env and resubmit:" >&2
  echo "  WITH_TRAINING=1 ENV_PREFIX=$CONDA_ENV sbatch scripts/dana_setup_env.sh" >&2
  exit 1
fi
cd "$REPO"

# The driver shells out to `python`, so the env's interpreter has to be first
# on PATH for the training processes it launches.
export PATH="$(dirname "$PY"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WORK

echo "host    $(hostname)"
echo "repo    $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "python  $PY"
echo "gpu     ${CUDA_VISIBLE_DEVICES:-none}"
# Only the GPU SLURM allocated. nvidia-smi otherwise lists every device on the
# node, which is harmless but misleading in a one-GPU job, and DANA's GPU
# policy is emphatic about not touching devices that were not allocated.
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" \
  --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

if [[ "$STAGES" == eval ]]; then
  echo "=== v6 tables, all seeds ==="
  STAGES=eval WORK="$WORK" "$DRIVER"
  echo "finished $(date -Is)"
  exit 0
fi

# --- no array: every seed x stem in sequence, in this one job ---------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "not an array job: training each of [$STEMS_ALL] for seeds [$SEEDS_ALL]"
  echo "in sequence on this GPU. One submitted job, which is what"
  echo "cscc-gpu-qos's 2-job cap leaves room for."
  echo "started $(date -Is)"
  STAGES=train SEEDS="$SEEDS_ALL" STEMS="$STEMS_ALL" PARALLEL=0 WORK="$WORK" \
    "$DRIVER"
  echo "finished $(date -Is)"
  exit 0
fi

# --- with --array: one (seed, stem) per task -------------------------------
read -r -a seeds <<< "$SEEDS_ALL"
read -r -a stems <<< "$STEMS_ALL"
n_stems=${#stems[@]}
idx="${SLURM_ARRAY_TASK_ID:-0}"
total=$(( ${#seeds[@]} * n_stems ))
if (( idx >= total )); then
  echo "task $idx is beyond ${#seeds[@]} seeds x $n_stems stems = $total; nothing to do."
  echo "Set --array=0-$((total - 1))%6 to match." >&2
  exit 0
fi
seed="${seeds[$(( idx / n_stems ))]}"
stem="${stems[$(( idx % n_stems ))]}"
echo "task $idx of $total  ->  seed $seed, stem $stem"
echo "started $(date -Is)"

# PARALLEL=0: this task owns one GPU and trains one stem on it. The driver's
# own parallelism is for a workstation with room for several at once.
STAGES=train SEEDS="$seed" STEMS="$stem" PARALLEL=0 WORK="$WORK" \
  "$DRIVER"

echo "finished $(date -Is)"
