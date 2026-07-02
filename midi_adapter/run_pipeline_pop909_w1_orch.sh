#!/usr/bin/env bash
# =============================================================================
#  Full POP909 pipeline — allow_wrong=1 filter + repair + transpose leadsheets
#  + AccoMontage Stage 2 (Option B) + song-level split + train.
#
#  Steps 1-3 and 5-7 run in the `rasp` conda env; step 4 (AccoMontage) runs
#  in the `sarr` env. The script picks up the right env for each step via
#  `conda run -n`, so you can execute it top-to-bottom in ANY active shell.
#
#  Re-running is safe: each step skips itself if its output already exists.
#  To force re-run, delete the corresponding output directory / file.
#
#  Usage:
#    bash run_pipeline_pop909_w1_orch.sh                # all steps
#    bash run_pipeline_pop909_w1_orch.sh 4              # only step 4
#    bash run_pipeline_pop909_w1_orch.sh 1 2 3          # only steps 1-3
# =============================================================================

set -euo pipefail

# ─── Paths (edit these) ──────────────────────────────────────────────────────
POP909=/l/users/xinyue.li/data/pop909_combined
ROOT=/l/users/xinyue.li/data/pop909_ivvi_w1
BASE_CKPT=/l/users/xinyue.li/rasp/checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
RASP_REPO=/l/users/xinyue.li/rasp
SA_REPO=/l/users/xinyue.li/Structured-Arrangement-Code         # ← edit
SA_DATA_ROOT_REL=data_file_dir                          # relative to SA_REPO

# ─── Derived paths (don't edit) ──────────────────────────────────────────────
MANIFEST=$ROOT/manifest.json
SAMPLES=$ROOT/samples
SNIPPETS=$ROOT/snippets
SNIPPETS_PERKEY=$ROOT/snippets_perkey
ORCH=$ROOT/orch_perkey
TRAIN_ALL=$ROOT/train_all_keys
VAL_ALL=$ROOT/val_all_keys
RUN_NAME=pop909_ivvi_w1_orch_perkey

# ─── Env activation helpers ──────────────────────────────────────────────────
# `conda run -n <env> --live-stream <cmd>` runs one command in the named env
# without needing conda to be initialized in the current shell.
rasp_run() { conda run -n rasp --live-stream --no-capture-output "$@"; }
sarr_run() { conda run -n sarr --live-stream --no-capture-output "$@"; }

# ─── Step selector ───────────────────────────────────────────────────────────
if [ $# -gt 0 ]; then
    STEPS=("$@")
else
    STEPS=(1 2 3 4 5 6 7)
fi
run_step() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

log() { echo -e "\n══════════════════════════════════════════════════════════════════\n▶ $*\n══════════════════════════════════════════════════════════════════"; }

mkdir -p "$ROOT"

# ─── Step 1: filter POP909 with allow_wrong=1 ────────────────────────────────
if run_step 1; then
    if [ -f "$MANIFEST" ]; then
        log "Step 1 — manifest already exists at $MANIFEST, skipping"
    else
        log "Step 1 — filter POP909 (allow_wrong=1, no adjacency)"
        cd "$RASP_REPO"
        rasp_run python -m midi_adapter.filter_nottingham filter \
            --input_dir         "$POP909" \
            --manifest          "$MANIFEST" \
            --n_bars 4 \
            --no_align \
            --allow_wrong 1 --max_consecutive_wrong 0 \
            --save_examples_dir "$SAMPLES" \
            --examples_per_key 1
    fi
fi

# ─── Step 2: snippet each matched window (repair applied per-track) ─────────
if run_step 2; then
    if [ -d "$SNIPPETS" ] && [ -n "$(ls -A "$SNIPPETS"/*.mid 2>/dev/null || true)" ]; then
        log "Step 2 — snippets already exist in $SNIPPETS, skipping"
    else
        log "Step 2 — write per-window multi-track snippets (with repair)"
        cd "$RASP_REPO"
        rasp_run python -m midi_adapter.filter_nottingham leadsheets \
            --manifest "$MANIFEST" \
            --out_dir  "$SNIPPETS" \
            --no_align
        echo "  snippet count: $(ls "$SNIPPETS"/*.mid | wc -l)"
    fi
fi

# ─── Step 3: pre-transpose to all 12 keys ────────────────────────────────────
if run_step 3; then
    if [ -d "$SNIPPETS_PERKEY" ] && [ -n "$(ls -A "$SNIPPETS_PERKEY"/*.mid 2>/dev/null || true)" ]; then
        log "Step 3 — transposed snippets already exist in $SNIPPETS_PERKEY, skipping"
    else
        log "Step 3 — transpose leadsheets into all 12 keys"
        cd "$RASP_REPO"
        rasp_run python -m midi_adapter.transpose_leadsheets \
            --in_dir      "$SNIPPETS" \
            --out_dir     "$SNIPPETS_PERKEY" \
            --target_keys 0 1 2 3 4 5 6 7 8 9 10 11
        echo "  transposed count: $(ls "$SNIPPETS_PERKEY"/*.mid | wc -l)"
    fi
fi

# ─── Step 4: AccoMontage Stage 2 (in the sarr conda env, from SA_REPO) ──────
if run_step 4; then
    if [ -d "$ORCH" ] && [ -n "$(find "$ORCH" -mindepth 2 -name 'arrangement_band-*.mid' -print -quit 2>/dev/null)" ]; then
        log "Step 4 — orchestrated output already present in $ORCH, skipping"
    else
        log "Step 4 — orchestrate every transposed snippet (Stage 2 only)"
        cd "$SA_REPO"
        sarr_run python "$RASP_REPO/midi_adapter/batch_orchestrate.py" \
            --in_dir    "$SNIPPETS_PERKEY" \
            --out_dir   "$ORCH" \
            --data_root "$SA_DATA_ROOT_REL/" \
            --n_bars 4 --tempo 120 --num_sample 2
    fi
fi

# ─── Step 5+6: extract CP tensors with rule filter + song-level split ───────
if run_step 5 || run_step 6; then
    if [ -f "${TRAIN_ALL}.pt" ] && [ -f "${VAL_ALL}.pt" ]; then
        log "Step 5+6 — extracted train/val .pt already exist, skipping"
    else
        log "Step 5+6 — extract CP tensors (rule_min_frac=0.75, song split 90/10)"
        cd "$RASP_REPO"
        rasp_run python -m midi_adapter.extract_orchestrated \
            --in_dir  "$ORCH" \
            --out_pt  "$TRAIN_ALL" \
            --max_polyphony 4 \
            --per_folder_key_only \
            --rule_min_frac 0.75 \
            --val_song_frac 0.1 --split_seed 42 --split train

        rasp_run python -m midi_adapter.extract_orchestrated \
            --in_dir  "$ORCH" \
            --out_pt  "$VAL_ALL" \
            --max_polyphony 4 \
            --per_folder_key_only \
            --rule_min_frac 0.75 \
            --val_song_frac 0.1 --split_seed 42 --split val
    fi

    # Split each output by key: seen (10 keys) vs unseen (F#, G#).
    if [ -f "${TRAIN_ALL}_seenkeys.pt" ] && [ -f "${VAL_ALL}_unseenkeys.pt" ]; then
        log "Step 6b — key-split .pt files already exist, skipping"
    else
        log "Step 6b — split each dataset by key (seen vs unseen)"
        rasp_run python - <<PY
import torch
SEEN   = {0, 1, 2, 3, 4, 5, 7, 9, 10, 11}
UNSEEN = {6, 8}
ROOT   = "$ROOT"

for name in ("train_all_keys", "val_all_keys"):
    prefix   = f"{ROOT}/{name}"
    data     = torch.load(f"{prefix}.pt",           weights_only=True)
    keys     = torch.load(f"{prefix}.keys.pt",      weights_only=True)
    csq      = torch.load(f"{prefix}.chord_seq.pt", weights_only=True)
    lengths  = torch.load(f"{prefix}.length.pt",    weights_only=True)
    W        = int(lengths[0])
    n        = len(keys)
    data3d   = data.view(n, W, -1)
    for label, ks in (("seenkeys", SEEN), ("unseenkeys", UNSEEN)):
        mask = torch.tensor([int(k) in ks for k in keys])
        out  = f"{prefix}_{label}"
        torch.save(data3d[mask].reshape(-1, data.shape[-1]),  f"{out}.pt")
        torch.save(lengths[mask],                             f"{out}.length.pt")
        torch.save(torch.zeros(int(mask.sum()), 2, dtype=torch.int8),
                                                              f"{out}.pitch_shift_range.pt")
        torch.save(keys[mask],                                f"{out}.keys.pt")
        torch.save(csq[mask],                                 f"{out}.chord_seq.pt")
        print(f"  {out}: {int(mask.sum())} windows")
PY
    fi
fi

# ─── Step 7: train ──────────────────────────────────────────────────────────
if run_step 7; then
    log "Step 7 — train adapter"
    cd "$RASP_REPO"
    rasp_run python -m midi_adapter.train_cp_yinyang \
        --base_ckpt   "$BASE_CKPT" \
        --train_data  "${TRAIN_ALL}_seenkeys.pt" \
        --val_data    "${VAL_ALL}_seenkeys.pt" \
        --unseen_data "${VAL_ALL}_unseenkeys.pt" \
        --approach chord --n_skip 1 --paired_chord_seq \
        --chords_per_bar 2 --model_size 1 --adapter_rank 256 --batch_size 8 \
        --max_steps 40000 \
        --run_name  "$RUN_NAME"
fi

log "Pipeline finished."
