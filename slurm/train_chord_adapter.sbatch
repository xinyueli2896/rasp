#!/bin/bash
# =============================================================================
#  Chord adapter, flagship config: no rule input, read before the stack,
#  compiled TracR program, CE + transposition-equivariance supervision.
#
#    sbatch slurm/train_chord_adapter.sbatch
#
#  Edit the settings block below to change the run. Everything in it is an
#  env var that midi_adapter/run_adapter_tracr.sh reads.
# =============================================================================
#SBATCH --job-name=chord-adapter
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
mkdir -p slurm_logs
source slurm/env.sh

# ── settings ────────────────────────────────────────────────────────────
export FROM_LAYER=0        # read the stream entering the stack
export PROGRAM=full        # compiled head + MLP; the program emits the root
export PROXY_W=1.0         # CE on the encoder's 12 root dims
export EQUIV_W=1.0         # transposition equivariance (JS form)
export MAX_STEPS=8000
# ────────────────────────────────────────────────────────────────────────

bash midi_adapter/run_adapter_tracr.sh direct
