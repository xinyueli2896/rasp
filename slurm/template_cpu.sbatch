#!/bin/bash
# =============================================================================
#  Copy this, put your command at the bottom, submit it.
#
#    cp slurm/template_cpu.sbatch slurm/my_job.sbatch
#    $EDITOR slurm/my_job.sbatch
#    sbatch slurm/my_job.sbatch
#
#  cscc-cpu-p + cscc-cpu-qos: 512 cores, 2 concurrent jobs, 72 h.
#  For the GPU-free work here -- POP909 filtering, snippet and CP-tensor
#  extraction, dataset merging, compiling a Tracr rule model. Every
#  dataloader runs num_workers=0, so most of these are single-threaded and
#  8-16 cores is plenty. Request only what you use.
#
#  Before any large-scale small-file work on /l/, mail hpc.admins@mbzuai.ac.ae
#  (policy). Staging intermediates in $TMPDIR and writing only .pt files to
#  /l/ is the mitigation that fits this pipeline.
# =============================================================================
#SBATCH --job-name=rasp-cpu
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
# sbatch runs this from SLURM_SUBMIT_DIR, i.e. wherever you ran sbatch, so
# submit from the repo root. Say so plainly rather than failing on a
# confusing "No such file" halfway down.
if [ ! -f slurm/env.sh ]; then
    echo "run sbatch from the repo root: cd <repo> && sbatch slurm/$(basename "$0")" >&2
    exit 1
fi
source slurm/env.sh

# ── your command ────────────────────────────────────────────────────────
python -m data.integer_dataset
