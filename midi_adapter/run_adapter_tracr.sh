#!/usr/bin/env bash
# =============================================================================
#  Compiled-rule-model adapter — the structural match to the integer experiment.
#
#  Every earlier chord run attends to a TABLE: ChordSeqRuleModel and
#  CPChordRuleModel are zero-parameter lookups that emit the answer directly,
#  so "the adapter reads a rule model" is really "the adapter reads the labels".
#  The integer experiment did something stronger — its proxy passed through a
#  compiled TracR transformer whose FROZEN attention did real computation.
#
#  ChordTracrRuleModel restores that. d_model = 28, laid out exactly as TracR:
#      dims  0-11  current chord root      (the proxy must supply this)
#      dims 12-23  retrieved tonic         (written by the frozen head)
#      dims 24-27  bar phase               (frozen clock, injected)
#
#  One frozen head implements
#      lookup = Select(phase, phase, lambda k, q: k == 0)
#      key    = Aggregate(lookup, root)
#  i.e. attend to every phase-0 slot and copy its root into the tonic subspace.
#  Verified exact: attention mass on phase-0 keys is 1.000000 for all 12 keys
#  and every query position, causal throughout.
#
#  That leaves  root = (key + OFFSETS[phase]) % 12  for the ADAPTER to compute.
#  The rule model retrieves; it does not solve — the same division of labour as
#  the integer experiment's seed_broadcast mode.
#
#  Because W_Q/W_K route only on the phase subspace and W_V/W_O move only the
#  root subspace, a proxy that fails to populate those cleanly yields a blurred
#  retrieval and a useless tonic. That is the constraint, in place of
#  --proxy_loss_weight, which stays at 0 here.
#
#  Runs:
#    pop909_ivvi_w1_adapter_bidir_tracr_direct
#    pop909_ivvi_w1_adapter_bidir_tracr_mixed
#
#  Usage:
#    bash midi_adapter/run_adapter_tracr.sh              # both runs + evals
#    bash midi_adapter/run_adapter_tracr.sh direct
#    PROXY_W=1.0 bash midi_adapter/run_adapter_tracr.sh  # belt AND braces
#    NO_POS=1 bash midi_adapter/run_adapter_tracr.sh     # make it learn phase too
#    HEADS=1  bash midi_adapter/run_adapter_tracr.sh     # single-source ablation
# =============================================================================

set -euo pipefail

RASP_REPO=/l/users/xinyue.li/rasp
D=/l/users/xinyue.li/data/pop909_ivvi_w1
BASE=checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
PROXY_W="${PROXY_W:-0}"
NO_POS="${NO_POS:-0}"
HEADS="${HEADS:-2}"

EXTRA=""; SUF=""
[ "$HEADS" != 1 ] && SUF="${SUF}_h${HEADS}"
[ "$PROXY_W" != 0 ] && EXTRA="$EXTRA --proxy_loss_weight $PROXY_W" && SUF="${SUF}_proxy${PROXY_W}"
[ "$NO_POS" = 1 ]   && EXTRA="$EXTRA --no_proxy_pos_inject"        && SUF="${SUF}_nopos"

# No --paired_chord_seq: the rule model receives no input. --rule_attention
# swaps the lookup for the compiled head.
COMMON_TRAIN="--base_ckpt $BASE \
    --approach chord --n_skip 1 --bidirectional --rule_attention --positional_qk \
    --rule_heads $HEADS \
    --chords_per_bar 2 --model_size 1 --adapter_rank 256 --batch_size 8 \
    --max_steps 40000 $EXTRA"

# --rule_attention MUST be repeated at eval: without it the rule model is
# rebuilt at width 16 instead of 28 and every ar_to_rule weight fails to load.
COMMON_EVAL="--base_ckpt $BASE \
    --approach chord --n_skip 1 --bidirectional --rule_attention --positional_qk \
    --rule_heads $HEADS --chords_per_bar 2 \
    --n_prompt_beats 16 --temperature 0 --save_n_per_key 3"
[ "$NO_POS" = 1 ] && COMMON_EVAL="$COMMON_EVAL --no_proxy_pos_inject"

cd "$RASP_REPO"
mkdir -p eval_logs_grid
WHICH="${1:-both}"

log() { echo -e "\n════════════════════════════════════════════\n▶ $*\n════════════════════════════════════════════"; }

best_ckpt() {
    ls "checkpoints/$1/"*.by_val_loss.*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1
}

run_one() {     # run_one <tag> <train-data flags...>
    local tag="$1"; shift
    local RUN="pop909_ivvi_w1_adapter_bidir_tracr_${tag}${SUF}"
    if [ -n "$(best_ckpt "$RUN")" ]; then
        log "$RUN — checkpoints exist, skipping training"
    else
        log "$RUN — training (compiled rule head, proxy_loss_weight=$PROXY_W)"
        python -m midi_adapter.train_cp_yinyang $COMMON_TRAIN "$@" --run_name "$RUN"
    fi

    local CKPT; CKPT=$(best_ckpt "$RUN")
    log "$RUN — evaluating $CKPT"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL --adapter_ckpt "$CKPT" \
        --seen_data "$D/val_all_keys_seenkeys.pt" \
        --unseen_data "$D/val_all_keys_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_tracr_${tag}${SUF}_orch/" \
        2>&1 | tee "eval_logs_grid/adapter_tracr_${tag}${SUF}_orch.log"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL --adapter_ckpt "$CKPT" \
        --seen_data "$D/direct_val_seenkeys.pt" \
        --unseen_data "$D/direct_val_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_tracr_${tag}${SUF}_direct/" \
        2>&1 | tee "eval_logs_grid/adapter_tracr_${tag}${SUF}_direct.log"
}

if [ "$WHICH" = both ] || [ "$WHICH" = direct ]; then
    run_one direct \
        --train_data  "$D/direct_train_seenkeys.pt" \
        --val_data    "$D/direct_val_seenkeys.pt" \
        --unseen_data "$D/direct_val_unseenkeys.pt"
fi

if [ "$WHICH" = both ] || [ "$WHICH" = mixed ]; then
    run_one mixed \
        --train_data    "$D/train_all_keys_seenkeys.pt" \
        --pretrain_data "$D/direct_train_seenkeys.pt" \
        --val_data      "$D/val_all_keys_seenkeys.pt" \
        --unseen_data   "$D/val_all_keys_unseenkeys.pt"
fi

log "Done. Logs in eval_logs_grid/adapter_tracr_*.log"
