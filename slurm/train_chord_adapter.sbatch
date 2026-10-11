#!/bin/bash
# =============================================================================
#  Chord adapter, flagship config: no rule input, read before the stack,
#  compiled TracR program, CE + transposition-equivariance supervision.
#
#    sbatch slurm/train_chord_adapter.sbatch
#
#  Edit the settings block below to change the run. Everything in it is an
#  env var that midi_adapter/run_adapter_tracr.sh reads.
#
#  WALL TIME is an estimate, not a measurement -- 8 h for MAX_STEPS=8000.
#  After the first run, get the real figure and trim it:
#      sacct -j <jobid> --format=JobID,JobName,Elapsed,State,ExitCode
#  Policy asks for only the runtime you need; over-requesting costs you
#  queue priority. The cap on long/gpu-12 is 72 h.
# =============================================================================
#SBATCH --job-name=chord-adapter
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --time=08:00:00   # 8k steps; raise for 40k (cap 72h on long)

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

# ── settings ────────────────────────────────────────────────────────────
export FROM_LAYER=0        # read the stream entering the stack
export PROGRAM=full        # compiled head + MLP; the program emits the root
export PROXY_W=1.0         # CE on the encoder's 12 root dims
export EQUIV_W=1.0         # transposition equivariance (JS form)
export MAX_STEPS=8000
# ────────────────────────────────────────────────────────────────────────

bash midi_adapter/run_adapter_tracr.sh direct
