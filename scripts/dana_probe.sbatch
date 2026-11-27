#!/bin/bash
# X11 round 1: cadence probes on the models WITHOUT an adapter.
#
#   models   base      the v0.42 checkpoint (BASE_CKPT)
#            finetune  the v6 Finetune baseline, seed FT_SEED (cp_bass_ft_v6_s<seed>)
#            random    random init, the probe's floor
#   probes   roman (I/IV/V) and next_chord (24), read just before each downbeat
#
# Smoke run first, on the debug QoS (100 spans per split, 3 layers, 1 seed):
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env SMOKE=1 sbatch --qos=gpu-debug-qos \
#       --time=00:30:00 -o $PWD/v6/logs_v6/probe-smoke-%j.out scripts/dana_probe.sbatch
#
# Then the real run, on the partition's production QoS (no --qos):
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env sbatch \
#       -o $PWD/v6/logs_v6/probe-%j.out scripts/dana_probe.sbatch
#
# One submitted job, so it leaves the other slot of the 2-job cap free.
# Output: $OUT/results.csv and $OUT/summary.csv.

#SBATCH --job-name=v6probe
#SBATCH --output=v6probe-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cscc-gpu-p
#SBATCH --time=08:00:00

set -euo pipefail

WORK="${WORK:?set WORK to the v6 work directory, e.g. \$PWD/v6}"
CONDA_ENV="${CONDA_ENV:-$WORK/env}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${PY:-$CONDA_ENV/bin/python}"
BASE_CKPT="${BASE_CKPT:-$WORK/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
DATA_PREFIX="${DATA_PREFIX:-$WORK/data/v6_chordal}"
FT_SEED="${FT_SEED:-1}"
FT_RUN="${FT_RUN:-$WORK/checkpoints_v6/cp_bass_ft_v6_s$FT_SEED}"
SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == 1 ]]; then
  OUT="${OUT:-$WORK/probe_r1_smoke}"
  EXTRACT_ARGS=(--max_spans 100)
  TRAIN_ARGS=(--layers 0,6,12 --seeds 0)
  [[ "${ALLOW_CPU:-0}" == 1 ]] && TRAIN_ARGS+=(--allow_cpu)
else
  OUT="${OUT:-$WORK/probe_r1}"
  EXTRACT_ARGS=()
  TRAIN_ARGS=()
fi

[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }
for f in "$BASE_CKPT" "${DATA_PREFIX}_finetune.pt" "${DATA_PREFIX}_unseen.pt"; do
  [[ -s "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
[[ -d "$FT_RUN" ]] || { echo "MISSING finetune run dir: $FT_RUN" >&2; exit 1; }
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

echo "host    $(hostname)"
echo "repo    $REPO ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "base    $BASE_CKPT"
echo "ft      $FT_RUN"
echo "out     $OUT   smoke=$SMOKE"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" \
  --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
# Stop now rather than after an hour of CPU extraction. ALLOW_CPU=1 overrides.
if [[ "${ALLOW_CPU:-0}" != 1 ]] && ! "$PY" -c "import sys, torch; sys.exit(not torch.cuda.is_available())"; then
  echo "no GPU visible to torch (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}');" >&2
  echo "check the allocation: sacct -j ${SLURM_JOB_ID:-<id>} -o JobID,Partition,QOS,AllocTRES%60" >&2
  exit 1
fi
echo "started $(date -Is)"

"$PY" -m midi_adapter.probe_extract \
    --data_prefix "$DATA_PREFIX" --out_dir "$OUT" \
    --model base="$BASE_CKPT" --model finetune="$FT_RUN" --model random=random:0 \
    "${EXTRACT_ARGS[@]}"

"$PY" -m midi_adapter.probe_train --probe_dir "$OUT" \
    --models base finetune random "${TRAIN_ARGS[@]}"

# Features are ~1 GB per model and rebuildable from the checkpoints; keep the
# CSVs, drop the features once the numbers are in (DANA data-management rule).
echo "finished $(date -Is)"
