#!/usr/bin/env bash
# =============================================================================
#  Generate MIDI from every trained model on BOTH test domains.
#
#  6 models x 2 test domains = 12 runs; each run internally splits the data
#  into seen-key / unseen-key halves, so you get 24 folders of audio:
#
#      gen_midi/<model>__<domain>/seen/    seen_00042_keyG.mid
#      gen_midi/<model>__<domain>/unseen/  unseen_00117_keyF.mid
#
#  The models, and the table row each one fills:
#
#    cell name                       table row                  checkpoint run
#    ------------------------------  -------------------------  -----------------------------
#    adapter_posqk_direct            FT w/ rule adapter . direct   adapter_posqk_direct
#    adapter_posqk_mixed             FT w/ rule adapter . mixed    adapter_posqk_mixed
#    adapter_posqk_content_direct    FT w/ adapter+LoRA . direct   adapter_posqk_content_direct
#    adapter_posqk_content_mixed     FT w/ adapter+LoRA . mixed    adapter_posqk_content_mixed
#    ft_direct                       FT w/o rule adapter . direct  ftbase_direct
#    ft_mixed                        FT w/o rule adapter . mixed   ftbase_mixed
#
#  NOTE: neither pos-QK variant carries LoRA — run_adapter_posqk_content.sh
#  trains with `--positional_qk --qk_content_residual` and no --lora_rank. The
#  row is filled by the content-residual variant per instruction; label it
#  "adapter + content-residual QK", not "+ LoRA", in any write-up.
#
#  Aliases so the old row names still work on the command line:
#    adapter_direct -> adapter_posqk_direct   lora_direct -> adapter_posqk_content_direct
#    adapter_mixed  -> adapter_posqk_mixed    lora_mixed  -> adapter_posqk_content_mixed
#
#  WITH_LEGACY=1 additionally generates from the pre-pos-QK checkpoints that
#  produced the current results table (direct_only_v3, orch_perkey_mixed,
#  adapter_lora16_posqk_*), under cell names prefixed `legacy_`.
#
#  ...and optionally BASE (the untuned pretrained transformer) as a reference
#  point for listening — enable with WITH_BASELINE=1.
#
#  Accuracies are still computed over EVERY window; SAVE_N only caps how many
#  MIDI files get written per key.
#
#  Usage:
#    bash midi_adapter/run_generate_midi.sh                  # all 6 models
#    bash midi_adapter/run_generate_midi.sh lora_mixed       # one model
#    bash midi_adapter/run_generate_midi.sh ft_direct ft_mixed
#    SAVE_N=10 TEMP=0.9 bash midi_adapter/run_generate_midi.sh
#    DOMAINS=direct bash midi_adapter/run_generate_midi.sh    # skip orch test
#    WITH_BASELINE=1 bash midi_adapter/run_generate_midi.sh
#
#  Env knobs:
#    SAVE_N=3        MIDI files saved per key per folder (0 = save all)
#    TEMP=0          0 = greedy/deterministic. Use 0.9 for varied listening.
#    DOMAINS="orch direct"
#    OUT=gen_midi    output root
#    FORCE=1         regenerate even if the folder already has .mid files
# =============================================================================

set -uo pipefail

RASP_REPO=/l/users/xinyue.li/rasp
D=/l/users/xinyue.li/data/pop909_ivvi_w1
BASE=checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt

SAVE_N="${SAVE_N:-3}"
TEMP="${TEMP:-0}"
DOMAINS="${DOMAINS:-orch direct}"
OUT="${OUT:-gen_midi}"
N_DEMOS="${N_DEMOS:-0}"
WITH_BASELINE="${WITH_BASELINE:-0}"
WITH_LEGACY="${WITH_LEGACY:-0}"
FORCE="${FORCE:-0}"

# Test sets. "orch" = orchestrated (AccoMontage) windows, "direct" = the
# directly-filtered POP909 windows. Both are held-out songs.
ORCH_SEEN=$D/val_all_keys_seenkeys.pt
ORCH_UNSEEN=$D/val_all_keys_unseenkeys.pt
DIR_SEEN=$D/direct_val_seenkeys.pt
DIR_UNSEEN=$D/direct_val_unseenkeys.pt

ADAPTER_FLAGS="--approach chord --n_skip 1 --paired_chord_seq --chords_per_bar 2"

# name | kind | checkpoint-dir (or '-' for BASE) | extra flags
#   kind=adapter -> ckpt goes to --adapter_ckpt, base stays frozen $BASE
#   kind=base    -> ckpt goes to --base_ckpt with --no_adapter (full finetune)
MODELS=(
  "adapter_posqk_direct|adapter|checkpoints/pop909_ivvi_w1_adapter_posqk_direct|$ADAPTER_FLAGS --positional_qk"
  "adapter_posqk_mixed|adapter|checkpoints/pop909_ivvi_w1_adapter_posqk_mixed|$ADAPTER_FLAGS --positional_qk"
  "adapter_posqk_content_direct|adapter|checkpoints/pop909_ivvi_w1_adapter_posqk_content_direct|$ADAPTER_FLAGS --positional_qk --qk_content_residual"
  "adapter_posqk_content_mixed|adapter|checkpoints/pop909_ivvi_w1_adapter_posqk_content_mixed|$ADAPTER_FLAGS --positional_qk --qk_content_residual"
  "ft_direct|base|checkpoints/pop909_ivvi_w1_ftbase_direct|"
  "ft_mixed|base|checkpoints/pop909_ivvi_w1_ftbase_mixed|"
)
[ "$WITH_BASELINE" = 1 ] && MODELS+=( "baseline|base|-|" )
# The checkpoints behind the current results table, kept reachable so the MIDI
# can be matched to the published numbers. The lora16 runs DO carry LoRA, so
# --lora_rank 16 is mandatory at eval or the LoRA weights load as zeros.
[ "$WITH_LEGACY" = 1 ] && MODELS+=(
  "legacy_adapter_direct|adapter|checkpoints/pop909_ivvi_w1_direct_only_v3|$ADAPTER_FLAGS"
  "legacy_adapter_mixed|adapter|checkpoints/pop909_ivvi_w1_orch_perkey_mixed|$ADAPTER_FLAGS"
  "legacy_lora_direct|adapter|checkpoints/pop909_ivvi_w1_adapter_lora16_posqk_direct|$ADAPTER_FLAGS --positional_qk --lora_rank 16"
  "legacy_lora_mixed|adapter|checkpoints/pop909_ivvi_w1_adapter_lora16_posqk_mixed|$ADAPTER_FLAGS --positional_qk --lora_rank 16"
)

cd "$RASP_REPO" || { echo "cannot cd $RASP_REPO"; exit 1; }
LOG_DIR="${OUT}_logs"
mkdir -p "$OUT" "$LOG_DIR"

log() { echo -e "\n════════════════════════════════════════════\n▶ $*\n════════════════════════════════════════════"; }

# Lowest-val_loss checkpoint in a run directory.
best_ckpt() {
    ls "$1/"*.by_val_loss.*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1
}

# Accept the old row names on the command line and map them onto the pos-QK
# checkpoints that now fill those rows.
canon() {
    case "$1" in
        adapter_direct) echo adapter_posqk_direct ;;
        adapter_mixed)  echo adapter_posqk_mixed ;;
        lora_direct)    echo adapter_posqk_content_direct ;;
        lora_mixed)     echo adapter_posqk_content_mixed ;;
        *)              echo "$1" ;;
    esac
}

WANTED=()
for a in "$@"; do WANTED+=( "$(canon "$a")" ); done
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

SKIPPED=()
for spec in "${MODELS[@]}"; do
    IFS='|' read -r NAME KIND CKPT_DIR EXTRA <<< "$spec"
    want "$NAME" || continue

    if [ "$CKPT_DIR" = '-' ]; then
        CKPT="$BASE"
    else
        CKPT="$(best_ckpt "$CKPT_DIR")"
    fi
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "!! $NAME — no checkpoint under $CKPT_DIR, skipping"
        SKIPPED+=("$NAME")
        continue
    fi

    if [ "$KIND" = adapter ]; then
        MODEL_ARGS="--base_ckpt $BASE --adapter_ckpt $CKPT $EXTRA"
    else
        MODEL_ARGS="--base_ckpt $CKPT --no_adapter"
    fi

    for DOM in $DOMAINS; do
        case "$DOM" in
            orch)   SEEN=$ORCH_SEEN; UNSEEN=$ORCH_UNSEEN ;;
            direct) SEEN=$DIR_SEEN;  UNSEEN=$DIR_UNSEEN  ;;
            *) echo "!! unknown domain '$DOM'"; continue ;;
        esac

        CELL="${NAME}__${DOM}"
        DEST="$OUT/$CELL"
        if [ "$FORCE" != 1 ] && [ -n "$(find "$DEST" -name '*.mid' 2>/dev/null | head -1)" ]; then
            log "$CELL — MIDI already present in $DEST, skipping (FORCE=1 to redo)"
            continue
        fi

        log "$CELL — ckpt $(basename "$CKPT")  |  temp=$TEMP  save_n_per_key=$SAVE_N"
        python -m midi_adapter.evaluate_on_real $MODEL_ARGS \
            --seen_data "$SEEN" --unseen_data "$UNSEEN" \
            --n_prompt_beats 16 --temperature "$TEMP" \
            --save_n_per_key "$SAVE_N" --n_demos_per_key "$N_DEMOS" \
            --save_midi_dir "$DEST/" \
            2>&1 | tee "$LOG_DIR/$CELL.log"
    done
done

log "MIDI written under $OUT/ — per-folder counts:"
find "$OUT" -name '*.mid' 2>/dev/null | awk -F/ '{NF--; print}' OFS=/ \
    | sort | uniq -c | sort -k2 || echo "  (none)"

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo -e "\nSkipped (checkpoint not found): ${SKIPPED[*]}"
fi
echo -e "\nLogs: $LOG_DIR/"
