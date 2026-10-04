#!/usr/bin/env bash
# =============================================================================
#  Probe a trained adapter WITHOUT retyping its flags.
#
#  The probe (and evaluate_on_real) must be given the same architecture flags
#  the checkpoint was trained with, or the modules are rebuilt at the wrong
#  shape and the weights silently fail to load. That has already produced two
#  wasted measurements: a width-28-vs-16 rule model from a missing
#  --rule_attention, and a whole table of random weights from a mistyped
#  checkpoint path (a missing ckpt is a WARNING, not an error, so the run looks
#  fine and means nothing).
#
#  run_adapter_tracr.sh encodes every one of those flags in the run directory
#  name, so derive them back from it instead of restating them.
#
#  Usage:
#    bash midi_adapter/probe_run.sh                 # list the runs it can probe
#    bash midi_adapter/probe_run.sh <run-dir-name>
#    bash midi_adapter/probe_run.sh <run-dir-name> --max_windows 400
#
#  Add DRY=1 to print the command instead of running it.
# =============================================================================

set -uo pipefail

RASP_REPO="${RASP_REPO:-/l/users/xinyue.li/rasp}"
D="${D:-/l/users/xinyue.li/data/pop909_ivvi_w1}"
BASE="${BASE:-checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt}"

cd "$RASP_REPO" || { echo "cannot cd $RASP_REPO"; exit 1; }

if [ $# -eq 0 ]; then
    echo "runs with a usable checkpoint:"
    found=0
    for d in checkpoints/*/; do
        n=$(basename "$d")
        c=$(ls "$d"*.by_val_loss.*.ckpt 2>/dev/null | wc -l)
        [ "$c" -eq 0 ] && continue
        found=1
        printf '  %-72s %2d ckpt\n' "$n" "$c"
    done
    [ "$found" = 0 ] && echo "  (none found under $RASP_REPO/checkpoints)"
    echo
    echo "then: bash midi_adapter/probe_run.sh <run-dir-name>"
    exit 0
fi

RUN="$1"; shift
RUN="${RUN%/}"; RUN="${RUN#checkpoints/}"
DIR="checkpoints/$RUN"
[ -d "$DIR" ] || { echo "no such run dir: $DIR"; echo "run with no arguments to list them."; exit 1; }

# Best by val_loss, same selection rule the runner uses.
CKPT=$(ls "$DIR"/*.by_val_loss.*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
[ -n "$CKPT" ] || { echo "no *.by_val_loss.*.ckpt in $DIR"; exit 1; }

# ── derive the architecture flags from the run name ──────────────────────
# Order matters in one place only: --rule_program mlp_only also contains the
# string "mlp", so match the numeric _mlp<N> (ar_to_rule_hidden) separately.
F=""
[[ "$RUN" == *_bidir_* ]]   && F="$F --bidirectional"
[[ "$RUN" == *_tracr* ]]    && F="$F --rule_attention"
[[ "$RUN" == *_nopos* ]]    && F="$F --no_proxy_pos_inject"
[[ "$RUN" == *_mlp_only* ]] && F="$F --rule_program mlp_only"
[[ "$RUN" == *_full* ]]     && F="$F --rule_program full"
[[ "$RUN" == *_triad* ]]    && F="$F --rule_input triad"
for act in sigmoid softmax hard; do
    [[ "$RUN" == *_${act}* ]] && F="$F --proxy_activation $act"
done
[[ "$RUN" =~ _L(-?[0-9]+) ]]   && F="$F --rule_from_layer ${BASH_REMATCH[1]}"
[[ "$RUN" =~ _mlp([0-9]+) ]]   && F="$F --ar_to_rule_hidden ${BASH_REMATCH[1]}"
[[ "$RUN" =~ _h([0-9]+) ]]     && F="$F --rule_heads ${BASH_REMATCH[1]}"
[[ "$RUN" =~ _cr([a-z]+) ]]    && F="$F --content_residual ${BASH_REMATCH[1]}"

# --positional_qk leaves no mark in the name: every tracr run sets it, and a
# run that did not would fail to load rather than mismeasure.
[[ "$RUN" == *_tracr* ]] && F="$F --positional_qk"

# Equivariance and proxy losses are TRAIN-time only and take no eval flag;
# _equiv*/_proxy* in the name are deliberately not translated.

echo "run   : $RUN"
echo "ckpt  : $CKPT"
echo "flags :$F"
echo

CMD=(python -m midi_adapter.probe_rule_proxy
     --base_ckpt "$BASE" --adapter_ckpt "$CKPT"
     --approach chord --n_skip 1 --chords_per_bar 2
     $F
     --data "$D/direct_val_seenkeys.pt" "$D/direct_val_unseenkeys.pt"
     "$@")

if [ "${DRY:-0}" = 1 ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
"${CMD[@]}"
