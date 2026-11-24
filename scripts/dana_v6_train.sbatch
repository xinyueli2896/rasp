#!/bin/bash
# Train the v6 stems on DANA's GPU partition, one array task per (seed, stem).
#
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env sbatch \
#       -o $PWD/v6/logs_v6/train-%A_%a.out scripts/dana_v6_train.sbatch
#
# 6 stems x 3 seeds = 18 tasks on cscc-gpu-p, throttled to 2 at a time
# because that partition allows 2 concurrent jobs and 4 GPUs per account, with
# a 24 h limit per job. Each task takes one GPU, so 18 tasks is 9 rounds --
# budget a day or two of wall-clock, and consider SEEDS=1 first (6 tasks, 3
# rounds) to get a table before spending 3x the time on the other two seeds.
#
# DANA's `long` partition would run 6 at a time (8 jobs, 12 GPUs, 72 h), which
# is three times faster for the same work. If you ever want it:
#   sbatch --partition=long --qos=gpu-12 --array=0-17%6 --time=48:00:00 ...
#
# An array rather than one long job because the stems are independent: a stem
# that dies costs its own task, not the run, and the queue can start them as
# GPUs free up instead of waiting for six at once.
#
#   SEEDS="1"              one seed (edit --array to match: stems - 1)
#   STEMS="frozen lora"    fewer stems (likewise)
#   MAX_STEPS=2000         a short smoke run before committing the full one
#
# PARTITION AND QOS. No QoS is requested in the directives, so the partition's
# own QoS applies: cscc-gpu-p carries cscc-gpu-qos. Pass one explicitly when
# you want the other: --qos=gpu-debug-qos for a smoke run, which is what DANA
# prescribes for environment and dependency checks.
#
# The eval is a separate submission once training is in: it needs every stem's
# checkpoint, so it cannot be an array task.
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env STAGES=eval sbatch \
#       -o $PWD/v6/logs_v6/eval-%j.out scripts/dana_v6_train.sbatch

#SBATCH --job-name=v6train
#SBATCH --output=v6train-%A_%a.out
#SBATCH --array=0-17%2
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [[ "$STAGES" == eval ]]; then
  echo "=== v6 tables, all seeds ==="
  STAGES=eval WORK="$WORK" ./scripts/run_v6_stems.sh
  echo "finished $(date -Is)"
  exit 0
fi

# --- one (seed, stem) per array task ---------------------------------------
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
  ./scripts/run_v6_stems.sh

echo "finished $(date -Is)"
