#!/usr/bin/env bash
# =============================================================================
#  "No rule input" adapter for the chord-root experiment.
#
#  Every other adapter run is HANDED the answer: --paired_chord_seq feeds the
#  8-slot chord sequence (or a key) straight into the rule model, so the model
#  only has to render a progression it was told. This variant is given nothing.
#
#      rule_hidden[t] = ar_to_rule(h[t])        # h = base's own hidden states
#
#  A learned Linear(768 -> 16) per adapter layer projects the FROZEN base's AR
#  hidden states into rule space; the cross-attention then reads K/V from that.
#  The base stack is causal and shifted by one, so proxy[t] depends only on
#  tokens < t. Nothing external is required at inference — the model must infer
#  the key from the prompt and apply I-IV-V-I itself.
#
#  --proxy_loss_weight is what makes this work. In the integer experiment the
#  proxy was pushed through the TracR rule model's FROZEN attention, which
#  pinned it to rule coordinates for free. CPChordRuleModel is a pure lookup
#  with no attention, so without an explicit target ar_to_rule is an
#  unconstrained Linear and the adapter collapses into plain self-attention
#  over music features. The aux loss supplies that constraint:
#      BCE(proxy[..., :12],  triad chromagram)  +  CE(proxy[..., 12:], bar phase)
#  Targets come from the dataset key at TRAIN time only.
#
#  Runs:
#    pop909_ivvi_w1_adapter_bidir_direct   direct-only data
#    pop909_ivvi_w1_adapter_bidir_mixed    mixed (direct + orchestrated)
#
#  Usage:
#    bash midi_adapter/run_adapter_bidir.sh            # both runs + evals
#    bash midi_adapter/run_adapter_bidir.sh direct
#    bash midi_adapter/run_adapter_bidir.sh mixed
#    PROXY_W=0 bash midi_adapter/run_adapter_bidir.sh  # ablate the aux loss
# =============================================================================

set -euo pipefail

RASP_REPO=/l/users/xinyue.li/rasp
D=/l/users/xinyue.li/data/pop909_ivvi_w1
BASE=checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
PROXY_W="${PROXY_W:-1.0}"

# NOTE: no --paired_chord_seq. That is the entire point — the rule model gets
# no input. --positional_qk still applies: with bidirectional, T_k == T_q, so
# query t is aligned to proxy t (key_stride = 1).
COMMON_TRAIN="--base_ckpt $BASE \
    --approach chord --n_skip 1 --bidirectional --positional_qk \
    --proxy_loss_weight $PROXY_W \
    --chords_per_bar 2 --model_size 1 --adapter_rank 256 --batch_size 8 \
    --max_steps 40000"

# --bidirectional MUST be repeated at eval or ar_to_rule is never constructed
# and the checkpoint's weights are silently dropped by the strict=False load.
COMMON_EVAL="--base_ckpt $BASE \
    --approach chord --n_skip 1 --bidirectional --positional_qk \
    --chords_per_bar 2 \
    --n_prompt_beats 16 --temperature 0 --save_n_per_key 3"

cd "$RASP_REPO"
mkdir -p eval_logs_grid
WHICH="${1:-both}"
SUF=""; [ "$PROXY_W" = 0 ] && SUF="_noproxy"

log() { echo -e "\n════════════════════════════════════════════\n▶ $*\n════════════════════════════════════════════"; }

best_ckpt() {   # best_ckpt <run_name> → path of lowest-val_loss ckpt
    ls "checkpoints/$1/"*.by_val_loss.*.ckpt 2>/dev/null \
        | sort -t= -k3 -g | head -1
}

run_one() {     # run_one <tag> <train-data flags...>
    local tag="$1"; shift
    local RUN="pop909_ivvi_w1_adapter_bidir_${tag}${SUF}"
    if [ -n "$(best_ckpt "$RUN")" ]; then
        log "$RUN — checkpoints exist, skipping training"
    else
        log "$RUN — training (no rule input, proxy_loss_weight=$PROXY_W)"
        python -m midi_adapter.train_cp_yinyang $COMMON_TRAIN "$@" --run_name "$RUN"
    fi

    local CKPT; CKPT=$(best_ckpt "$RUN")
    log "$RUN — evaluating $CKPT"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL --adapter_ckpt "$CKPT" \
        --seen_data "$D/val_all_keys_seenkeys.pt" \
        --unseen_data "$D/val_all_keys_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_bidir_${tag}${SUF}_orch/" \
        2>&1 | tee "eval_logs_grid/adapter_bidir_${tag}${SUF}_orch.log"
    python -m midi_adapter.evaluate_on_real $COMMON_EVAL --adapter_ckpt "$CKPT" \
        --seen_data "$D/direct_val_seenkeys.pt" \
        --unseen_data "$D/direct_val_unseenkeys.pt" \
        --save_midi_dir "eval_midi/adapter_bidir_${tag}${SUF}_direct/" \
        2>&1 | tee "eval_logs_grid/adapter_bidir_${tag}${SUF}_direct.log"
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

log "Done. Logs in eval_logs_grid/adapter_bidir_*.log"
