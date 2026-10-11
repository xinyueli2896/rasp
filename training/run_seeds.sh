#!/usr/bin/env bash
# =============================================================================
#  Seed sweep for the integer experiment.
#
#    bash training/run_seeds.sh                 # all three rows, seeds 0-4
#    bash training/run_seeds.sh finetune        # just the FT row
#    bash training/run_seeds.sh pretrain finetune
#    SEEDS="0 1 2" bash training/run_seeds.sh finetune
#
#  Rows:
#    pretrain   AR on pool A                       -> the baseline
#    finetune   AR continued on pool B, no adapter -> forgetting control
#    adapter    cross-attention over the frozen AR -> the proposed model
#
#  WHY EACH FT SEED NEEDS ITS OWN AR
#  ---------------------------------
#  --ar_ckpt defaults to seed0, so a sweep that omits it trains five
#  fine-tunes over ONE base. The spread then measures only fine-tuning
#  randomness (data order, dropout) and hides pretraining variance, which is
#  the larger source here -- the AR memorises five starters, and which
#  solution it lands in differs by seed. This script pairs seed N with
#  ar_integer_seed<N>.pt, so `pretrain` has to have been run for the same
#  seeds first.
#
#  Then:  python -m training.results_table
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

SEEDS="${SEEDS:-0 1 2 3 4}"
ROWS=("$@"); [ ${#ROWS[@]} -eq 0 ] && ROWS=(pretrain finetune adapter)
want() { for r in "${ROWS[@]}"; do [ "$r" = "$1" ] && return 0; done; return 1; }

for s in $SEEDS; do
    AR="checkpoints/ar_integer_seed${s}.pt"

    if want pretrain; then
        echo; echo "══ seed $s: AR pretrain on pool A ══"
        python -m training.pretrain_integer --seed "$s" || exit 1
    fi

    if want finetune || want adapter; then
        if [ ! -f "$AR" ]; then
            echo "missing $AR -- run the pretrain row for seed $s first:" >&2
            echo "  python -m training.pretrain_integer --seed $s" >&2
            exit 1
        fi
    fi

    if want finetune; then
        echo; echo "══ seed $s: FT on pool B, no adapter (forgetting control) ══"
        python -m training.finetune_adapter_integer --mode finetune \
            --seed "$s" --ar_ckpt "$AR" || exit 1
    fi

    if want adapter; then
        echo; echo "══ seed $s: adapter over the frozen AR ══"
        python -m training.finetune_adapter_integer --mode adapter \
            --seed "$s" --ar_ckpt "$AR" || exit 1
    fi
done

echo; echo "done. Build the table with:  python -m training.results_table"
