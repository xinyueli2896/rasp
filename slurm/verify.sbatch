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
#  Edit the #SBATCH lines below to match your cluster -- partition names,
#  account strings and GPU gres syntax differ everywhere, so nothing here can
#  be right by default.
# =============================================================================
#SBATCH --job-name=rasp-verify
#SBATCH --output=slurm_logs/verify-%j.out
#SBATCH --error=slurm_logs/verify-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

ENV_NAME="${ENV_NAME:-rasp}"

echo "node      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-<none>}"
echo "gpus      : ${CUDA_VISIBLE_DEVICES:-<none>}"
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
