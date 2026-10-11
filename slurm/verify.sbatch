#!/usr/bin/env bash
# =============================================================================
#  Verify the environment ON A COMPUTE NODE.
#
#  verify_env.py builds the 112M CP transformer and runs a forward pass. That
#  is exactly the kind of thing login nodes are not for, so it is a job.
#
#    sbatch slurm/verify.sbatch
#    # or with your cluster's settings:
#    SB_PARTITION=gpu SB_ACCOUNT=myproj sbatch slurm/verify.sbatch
#
#  MBZUAI DANA policy: interactive and debugging workloads -- environment
#  testing, dependency validation, GPU verification -- must use
#  --qos=gpu-debug-qos. That is precisely what this job is, so it is set
#  below. Production training must NOT use this QoS.
#
#  Partition and QoS are PAIRED on DANA -- gpu-debug-qos belongs to
#  cscc-gpu-p, not to `long`. The documented matrix:
#      cscc-gpu-p + cscc-gpu-qos   standard GPU work
#      cscc-gpu-p + gpu-debug-qos  interactive / debugging   <- this job
#      long       + gpu-12         large & general GPU work
#      cscc-cpu-p + cscc-cpu-qos   CPU-only work
#  Pairing them wrongly gets the job rejected.
# =============================================================================
#SBATCH --job-name=rasp-verify
#SBATCH --output=slurm_logs/verify-%j.out
#SBATCH --error=slurm_logs/verify-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=gpu-debug-qos

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

ENV_NAME="${ENV_NAME:-rasp}"

echo "node      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-<none>}"
# Printed only. Policy forbids SETTING this or otherwise reaching for GPUs
# Slurm did not allocate; nothing in this repo does.
echo "gpus      : ${CUDA_VISIBLE_DEVICES:-<none>}  (read-only)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
    echo "nvidia-smi unavailable"
echo

# Activate however this cluster does it. `conda activate` needs the shell hook,
# which a non-interactive job does not get from .bashrc.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" && conda activate "$ENV_NAME"
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python verify_env.py
