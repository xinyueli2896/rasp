#!/usr/bin/env bash
# =============================================================================
#  Run everything directly on an interactive multi-GPU node -- no Slurm.
#
#    bash run_local.sh            # rule model + data config + env check
#    bash run_local.sh quick      # skip the 112M forward pass
#
#  On a node with several GPUs, pass --devices 1 to the trainers. The default
#  (-1 = all) switches Lightning to DDP, and this repo does not benefit:
#  the model is 112M at batch_size 8, and the dataloaders are IterableDataset
#  with no distributed sampler, so extra ranks duplicate work instead of
#  splitting it. Scale with --batch_size, not --devices.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

echo "═══ GPUs ═══"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "no nvidia-smi"
echo

echo "═══ rule model ═══"
python -m models.integer_rule_model --max_seq_len 32 || exit 1
echo
echo "═══ data config ═══"
python -m data.integer_dataset || exit 1
echo
for mode in first every_cycle; do
    echo "═══ rule-model tests: seed_mode=$mode ═══"
    python -m models.test_integer_rule_model --max_seq_len 32 --seed_mode "$mode" || exit 1
    echo
done

echo "═══ environment ═══"
if [ "${1:-}" = quick ]; then
    python verify_env.py --quick
else
    python verify_env.py
fi
