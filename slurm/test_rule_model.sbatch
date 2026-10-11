#!/bin/bash
# =============================================================================
#  Test the Tracr-compiled integer rule model.
#
#    sbatch slurm/test_rule_model.sbatch
#
#  CPU partition on purpose. The model is 75-dimensional with no trainable
#  parameters, the weights are committed under models/compiled_rules/, and
#  the whole suite is seconds of arithmetic -- taking one of your 12 GPUs
#  for it would waste it, and policy asks for only the resources a workload
#  actually needs.
#
#  Needs only torch + numpy. tracr/jax are NOT required: section 1 of the
#  suite (lift-vs-Tracr) reports SKIP without them and the other 31 checks
#  run from the committed weights.
# =============================================================================
#SBATCH --job-name=rule-model-test
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
if [ ! -f slurm/env.sh ]; then
    echo "run sbatch from the repo root: cd <repo> && sbatch slurm/$(basename "$0")" >&2
    exit 1
fi
source slurm/env.sh

echo "═══ committed rule-model weights ═══"
ls -la models/compiled_rules/ 2>/dev/null || echo "MISSING — did git pull succeed?"
echo

echo "═══ the model, and what its hidden states hold ═══"
python -m models.integer_rule_model --max_seq_len 32 || exit 1
echo

echo "═══ data config (chord-matched split) ═══"
python -m data.integer_dataset || exit 1
echo

for mode in first every_cycle; do
    echo "═══ test suite: seed_mode=$mode ═══"
    python -m models.test_integer_rule_model --max_seq_len 32 --seed_mode "$mode" || exit 1
    echo
done

echo "all rule-model checks passed."
