#!/bin/bash
# =============================================================================
#  Copy this, put your command at the bottom, submit it.
#
#    cp slurm/template_cpu.sbatch slurm/my_job.sbatch
#    $EDITOR slurm/my_job.sbatch
#    sbatch slurm/my_job.sbatch
#
#  long + gpu-12: 12 GPUs concurrently, 8 jobs, 72 h, minimum 1 GPU.
#
#  gpu-debug-qos is NOT valid here -- it pairs with cscc-gpu-p. For short
#  tests use slurm/verify.sbatch's pairing instead.
#
#  MULTI-GPU: add --ntasks-per-node equal to the GPU count AND prefix the
#  command with `srun`. Both trainers use devices=-1 and Lightning expects
#  one task per GPU; bare `python` with 4 GPUs visible gives one process
#  that hangs or silently uses one device.
#      #SBATCH --gres=gpu:4
#      #SBATCH --ntasks-per-node=4
#      #SBATCH --cpus-per-task=16
#      #SBATCH --mem=230G
#      srun python -m ...
# =============================================================================
#SBATCH --job-name=rasp-gpu
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
mkdir -p slurm_logs
source slurm/env.sh

# ── your command ────────────────────────────────────────────────────────
python -m models.test_integer_rule_model --max_seq_len 32
