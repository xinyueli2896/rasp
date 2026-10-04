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
#    FROM_LAYER=0 bash midi_adapter/run_adapter_tracr.sh # ONE shared rule signal,
#                                                        # read before the stack
#    FROM_LAYER=1 bash midi_adapter/run_adapter_tracr.sh # ONE, after layer 1
#    PROGRAM=mlp_only PROXY_ACT=hard \
#      bash midi_adapter/run_adapter_tracr.sh            # no head: the proxy
#                                                        # estimates the key itself
#    PROGRAM=full RULE_IN=triad PROXY_ACT=hard \
#      bash midi_adapter/run_adapter_tracr.sh            # conventional compile,
#                                                        # chromagram in/out
#    PROGRAM=full bash midi_adapter/run_adapter_tracr.sh # conventional TracR:
#                                                        # head + MLP, the program
#                                                        # emits the root itself
#    CR=q bash midi_adapter/run_adapter_tracr.sh         # music content in Q,
#                                                        # keys stay pure addresses
# =============================================================================

set -euo pipefail

RASP_REPO=/l/users/xinyue.li/rasp
D=/l/users/xinyue.li/data/pop909_ivvi_w1
BASE=checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
PROXY_W="${PROXY_W:-0}"
NO_POS="${NO_POS:-0}"
HEADS="${HEADS:-2}"
FROM_LAYER="${FROM_LAYER:--1}"
CR="${CR:-none}"
# Programs with an MLP need the key region bounded in [0,1]; sigmoid is
# the right default when dims 0-11 are read as a chromagram.
if [ "${PROGRAM:-retrieve}" != retrieve ]; then
    PROXY_ACT="${PROXY_ACT:-sigmoid}"
else
    PROXY_ACT="${PROXY_ACT:-none}"
fi
PROGRAM="${PROGRAM:-retrieve}"
RULE_IN="${RULE_IN:-root}"

EXTRA=""; SUF=""
# rule_heads only applies to the retrieve program: full uses one head
# and mlp_only none, so tagging the run _h2 there would be a lie.
[ "$PROGRAM" = retrieve ] && [ "$HEADS" != 1 ] && SUF="${SUF}_h${HEADS}"
[ "$PROGRAM" = retrieve ] || HEADS=1
[ "$FROM_LAYER" != -1 ] && EXTRA="$EXTRA --rule_from_layer $FROM_LAYER" && SUF="${SUF}_L${FROM_LAYER}"
[ "$CR" != none ] && EXTRA="$EXTRA --content_residual $CR" && SUF="${SUF}_cr${CR}"
[ "$PROXY_ACT" != none ] && EXTRA="$EXTRA --proxy_activation $PROXY_ACT" && SUF="${SUF}_${PROXY_ACT}"
[ "$PROGRAM" != retrieve ] && EXTRA="$EXTRA --rule_program $PROGRAM" && SUF="${SUF}_${PROGRAM}"
[ "$RULE_IN" != root ] && EXTRA="$EXTRA --rule_input $RULE_IN" && SUF="${SUF}_${RULE_IN}"
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
    $( [ "$FROM_LAYER" != -1 ] && echo "--rule_from_layer $FROM_LAYER" ) \
    $( [ "$CR" != none ] && echo "--content_residual $CR" ) \
    $( [ "$PROXY_ACT" != none ] && echo "--proxy_activation $PROXY_ACT" ) \
    $( [ "$PROGRAM" != retrieve ] && echo "--rule_program $PROGRAM" ) \
    $( [ "$RULE_IN" != root ] && echo "--rule_input $RULE_IN" ) \
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
