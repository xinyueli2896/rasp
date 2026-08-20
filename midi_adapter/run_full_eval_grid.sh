#!/usr/bin/env bash
# =============================================================================
#  Run the COMPLETE evaluation grid — 5 models × 2 test domains — and print
#  the final summary tables (halfbar_acc POOL mean ± std, seen & unseen keys).
#
#  Models : baseline / adapter-direct(v3) / adapter-mixed / ft-direct / ft-mixed
#  Domains: orchestrated val set / direct val set   (same 7 held-out songs)
#
#  All evals run at temperature 0 (greedy) → fully deterministic, safe to
#  re-run. Per-cell logs land in $LOG_DIR; generated MIDIs in eval_midi/.
#
#  Usage:  bash midi_adapter/run_full_eval_grid.sh
# =============================================================================

set -euo pipefail

# ─── Paths (edit checkpoints here when models are retrained) ─────────────────
RASP_REPO=/l/users/xinyue.li/rasp
D=/l/users/xinyue.li/data/pop909_ivvi_w1

BASE=checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
ADAPTER_DIRECT='checkpoints/pop909_ivvi_w1_direct_only_v3/pop909_ivvi_w1_direct_only_v3.by_val_loss.epoch=00.val_loss=1.58653.ckpt'
ADAPTER_MIXED='checkpoints/pop909_ivvi_w1_orch_perkey_mixed/pop909_ivvi_w1_orch_perkey_mixed.by_val_loss.epoch=00.val_loss=1.55711.ckpt'
FT_DIRECT='checkpoints/pop909_ivvi_w1_ftbase_direct/pop909_ivvi_w1_ftbase_direct.by_val_loss.epoch=00.val_loss=1.14533.ckpt'
FT_MIXED='checkpoints/pop909_ivvi_w1_ftbase_mixed/pop909_ivvi_w1_ftbase_mixed.by_val_loss.epoch=00.val_loss=1.30369.ckpt'

ORCH_SEEN=$D/val_all_keys_seenkeys.pt
ORCH_UNSEEN=$D/val_all_keys_unseenkeys.pt
DIR_SEEN=$D/direct_val_seenkeys.pt
DIR_UNSEEN=$D/direct_val_unseenkeys.pt

ADAPTER_FLAGS="--approach chord --n_skip 1 --paired_chord_seq --chords_per_bar 2"
EVAL_FLAGS="--n_prompt_beats 16 --temperature 0 --save_n_per_key 3"

LOG_DIR=eval_logs_grid
cd "$RASP_REPO"
mkdir -p "$LOG_DIR"

# Master log: everything this script prints (all cell outputs + final grid)
# is mirrored into one file, in addition to the per-cell logs.
exec > >(tee "$LOG_DIR/full_run.log") 2>&1

log() { echo -e "\n════════════════════════════════════════════\n▶ $*\n════════════════════════════════════════════"; }

run_eval() {   # run_eval <cell_name> <extra model args...>
    local cell="$1"; shift
    local seen unseen
    case "$cell" in
        *_orch)   seen=$ORCH_SEEN; unseen=$ORCH_UNSEEN ;;
        *_direct) seen=$DIR_SEEN;  unseen=$DIR_UNSEEN  ;;
    esac
    if [ -s "$LOG_DIR/$cell.log" ] && grep -q 'POOL' "$LOG_DIR/$cell.log"; then
        log "$cell — log already complete, skipping (delete $LOG_DIR/$cell.log to re-run)"
        return
    fi
    log "$cell"
    python -m midi_adapter.evaluate_on_real "$@" \
        --seen_data "$seen" --unseen_data "$unseen" \
        $EVAL_FLAGS --save_midi_dir "eval_midi/$cell/" \
        2>&1 | tee "$LOG_DIR/$cell.log"
}

# ═══ The 10 cells ════════════════════════════════════════════════════════════
run_eval baseline_orch        --base_ckpt "$BASE" --no_adapter
run_eval baseline_direct      --base_ckpt "$BASE" --no_adapter
run_eval adapter_direct_orch    --base_ckpt "$BASE" --adapter_ckpt "$ADAPTER_DIRECT" $ADAPTER_FLAGS
run_eval adapter_direct_direct  --base_ckpt "$BASE" --adapter_ckpt "$ADAPTER_DIRECT" $ADAPTER_FLAGS
run_eval adapter_mixed_orch     --base_ckpt "$BASE" --adapter_ckpt "$ADAPTER_MIXED" $ADAPTER_FLAGS
run_eval adapter_mixed_direct   --base_ckpt "$BASE" --adapter_ckpt "$ADAPTER_MIXED" $ADAPTER_FLAGS
run_eval ft_direct_orch       --base_ckpt "$FT_DIRECT" --no_adapter
run_eval ft_direct_direct     --base_ckpt "$FT_DIRECT" --no_adapter
run_eval ft_mixed_orch        --base_ckpt "$FT_MIXED" --no_adapter
run_eval ft_mixed_direct      --base_ckpt "$FT_MIXED" --no_adapter

# ═══ Summary: parse POOL rows out of the logs and print the grid ═════════════
log "FINAL GRID"
python - "$LOG_DIR" <<'PY'
import os, re, sys

log_dir = sys.argv[1]
MODELS  = ['baseline', 'adapter_direct', 'adapter_mixed', 'ft_direct', 'ft_mixed']
DOMAINS = ['orch', 'direct']

def parse(cell):
    """Return {(SEEN|UNSEEN): (mean, std, n)} for halfbar_acc POOL rows."""
    path = os.path.join(log_dir, f'{cell}.log')
    if not os.path.exists(path):
        return {}
    out, section = {}, None
    for line in open(path):
        if '── SEEN keys' in line:
            section = 'SEEN'
        elif '── UNSEEN keys' in line:
            section = 'UNSEEN'
        elif section and line.strip().startswith('POOL'):
            m = re.match(r'\s*POOL\s+(\d+)\s+([\d.]+)\s+±([\d.]+)', line)
            if m and section not in out:
                out[section] = (float(m.group(2)), float(m.group(3)), int(m.group(1)))
    return out

for keyset in ('SEEN', 'UNSEEN'):
    print(f'\n  halfbar_acc — {keyset} keys  (POOL mean ± std [n])')
    print(f'  {"model":<16}  {"orchestrated test":>24}  {"direct test":>24}')
    for model in MODELS:
        row = [f'  {model:<16}']
        for dom in DOMAINS:
            r = parse(f'{model}_{dom}').get(keyset)
            row.append(f'{r[0]:.3f} ±{r[1]:.3f} [{r[2]:>3}]'.rjust(24) if r else 'MISSING'.rjust(24))
        print('  '.join(row))
print()
PY

log "Done. Logs in $LOG_DIR/, MIDIs in eval_midi/, grid above."
