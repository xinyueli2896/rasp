#!/usr/bin/env bash
# =============================================================================
#  Survey, then optionally reclaim, workspace space.
#
#  DRY RUN BY DEFAULT. Nothing is deleted unless you pass --delete.
#
#  What it NEVER touches:
#    * the pretrained base checkpoint
#    * anything under the data directory
#    * git-tracked files  (there are none under checkpoints/ or the log dirs)
#
#  What it offers to remove:
#    ckpt-extra   checkpoints in a run dir other than its best by_val_loss and
#                 best by_unseen_acc
#    midi         eval_midi*/ and gen_midi*/ — regenerable from checkpoints
#    wandb        wandb/ run logs
#    runs         entire run directories you name in STALE_RUNS
#    logs         eval_logs_grid/*.log. NOT in the default set, because these
#                 are results -- but every adapter log predating the sampling
#                 fix is STALE, and run_full_eval_grid.sh REPLAYS a cached log
#                 instead of re-running that cell. Leaving them in place keeps
#                 the old numbers alive. Delete them before re-running the grid.
#
#  Usage:
#    bash cleanup_workspace.sh                    # survey only
#    bash cleanup_workspace.sh --delete midi      # just the MIDI
#    bash cleanup_workspace.sh --delete ckpt-extra midi wandb
#    STALE_RUNS="old_run_v1 old_run_v2" bash cleanup_workspace.sh --delete runs
# =============================================================================

set -uo pipefail

REPO="${REPO:-/l/users/xinyue.li/rasp}"
DATA="${DATA:-/l/users/xinyue.li/data/pop909_ivvi_w1}"
BASE_CKPT_GLOB='cp_transformer_v*.ckpt'
STALE_RUNS="${STALE_RUNS:-}"
# Runs to protect from `wipe-runs`. Full fine-tunes cost ~40k steps each and
# are UNAFFECTED by the sampling fix (they evaluate through --no_adapter), so
# they stay valid comparison rows. Override with KEEP_RUNS=... to change.
KEEP_RUNS="${KEEP_RUNS:-pop909_ivvi_w1_ftbase_direct pop909_ivvi_w1_ftbase_mixed}"

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

DELETE=0; declare -a TARGETS=()
for a in "$@"; do
    case "$a" in
        --delete) DELETE=1 ;;
        *) TARGETS+=("$a") ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(ckpt-extra midi wandb runs)
# wipe-runs supersedes ckpt-extra: no point pruning inside a dir we delete.
want_wipe() { for t in "${TARGETS[@]}"; do [ "$t" = wipe-runs ] && return 0; done; return 1; }

want() { for t in "${TARGETS[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }
hsize() { du -sh "$@" 2>/dev/null | awk '{s=$1} END{print s?s:"0"}'; }
bytes() { du -sc "$@" 2>/dev/null | tail -1 | awk '{print $1+0}'; }

echo "repo: $REPO"
echo "data: $DATA   (never touched)"
df -h . | tail -1 | awk '{printf "disk: %s used of %s, %s available\n", $3, $2, $4}'
echo

# ── what exists ─────────────────────────────────────────────────────────
echo "═══ current usage ═══"
for d in checkpoints eval_midi gen_midi eval_logs_grid wandb lightning_logs; do
    [ -d "$d" ] && printf '  %-18s %s\n' "$d" "$(hsize "$d")"
done
[ -d "$DATA" ] && printf '  %-18s %s   (keep)\n' "$(basename "$DATA")" "$(hsize "$DATA")"
echo

# ── the keep list ───────────────────────────────────────────────────────
echo "═══ checkpoints: keeping the best of each run ═══"
KEEP=$(mktemp); DROP=$(mktemp)
find checkpoints -maxdepth 1 -name "$BASE_CKPT_GLOB" >> "$KEEP" 2>/dev/null
for run in checkpoints/*/; do
    [ -d "$run" ] || continue
    name=$(basename "$run")
    if [ -n "$STALE_RUNS" ] && grep -qw "$name" <<<"$STALE_RUNS"; then
        continue                           # handled by the `runs` target
    fi
    best_loss=$(ls "$run"*.by_val_loss.*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
    best_acc=$(ls "$run"*.by_unseen_acc.*.ckpt 2>/dev/null | sort -t= -k3 -gr | head -1)
    [ -n "$best_loss" ] && echo "$best_loss" >> "$KEEP"
    [ -n "$best_acc" ]  && echo "$best_acc"  >> "$KEEP"
    n_all=$(ls "$run"*.ckpt 2>/dev/null | wc -l)
    n_keep=$(( $([ -n "$best_loss" ] && echo 1 || echo 0) + $([ -n "$best_acc" ] && echo 1 || echo 0) ))
    printf '  %-52s %3d ckpt, keeping %d  (%s)\n' "$name" "$n_all" "$n_keep" "$(hsize "$run")"
    ls "$run"*.ckpt 2>/dev/null | grep -vxF -f "$KEEP" >> "$DROP"
done
echo

# ── build the removal set ───────────────────────────────────────────────
declare -a PLAN=()
add() { [ -e "$1" ] && PLAN+=("$1"); }

if want wipe-runs; then
    echo "═══ wipe-runs: every run dir except KEEP_RUNS ═══"
    echo "  protecting: ${KEEP_RUNS:-<none>}"
    for run in checkpoints/*/; do
        [ -d "$run" ] || continue
        name=$(basename "$run")
        if [ -n "$KEEP_RUNS" ] && grep -qw "$name" <<<"$KEEP_RUNS"; then
            printf '  %-54s %8s  KEEP\n' "$name" "$(hsize "$run")"
        else
            printf '  %-54s %8s  drop\n' "$name" "$(hsize "$run")"
            add "$run"
        fi
    done
    echo
fi
if want ckpt-extra && ! want_wipe && [ -s "$DROP" ]; then
    echo "═══ ckpt-extra: $(wc -l < "$DROP") non-best checkpoints, $(bytes $(cat "$DROP")) KB ═══"
    head -5 "$DROP" | sed 's/^/  /'
    [ "$(wc -l < "$DROP")" -gt 5 ] && echo "  ... and $(( $(wc -l < "$DROP") - 5 )) more"
    while read -r f; do add "$f"; done < "$DROP"
    echo
fi
if want midi; then
    for d in eval_midi gen_midi eval_midi_* gen_midi_* decoded_midi; do
        [ -d "$d" ] && { echo "  midi    $d  $(hsize "$d")"; add "$d"; }
    done
fi
if want logs; then
    n=$(ls eval_logs_grid/*.log 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  logs    eval_logs_grid/ ($n log files, $(hsize eval_logs_grid))"
        echo "          -> the grid script replays cached logs; removing these"
        echo "             forces every cell to actually re-run"
        while read -r f; do add "$f"; done < <(ls eval_logs_grid/*.log)
    fi
fi
if want wandb; then
    for d in wandb lightning_logs; do
        [ -d "$d" ] && { echo "  logs    $d  $(hsize "$d")"; add "$d"; }
    done
fi
if want runs && [ -n "$STALE_RUNS" ]; then
    for r in $STALE_RUNS; do
        [ -d "checkpoints/$r" ] && { echo "  run     checkpoints/$r  $(hsize "checkpoints/$r")"; add "checkpoints/$r"; }
    done
fi

echo
if [ ${#PLAN[@]} -eq 0 ]; then
    echo "nothing selected to remove."
    rm -f "$KEEP" "$DROP"; exit 0
fi
echo "═══ would reclaim $(bytes "${PLAN[@]}") KB across ${#PLAN[@]} paths ═══"

# ── safety: nothing git-tracked, nothing under data, no base ckpt ───────
BAD=0
for p in "${PLAN[@]}"; do
    case "$p" in "$DATA"*) echo "REFUSING (data): $p"; BAD=1 ;; esac
    case "$(basename "$p")" in cp_transformer_v*) echo "REFUSING (base ckpt): $p"; BAD=1 ;; esac
    git ls-files --error-unmatch "$p" >/dev/null 2>&1 && { echo "REFUSING (git-tracked): $p"; BAD=1; }
done
[ "$BAD" = 1 ] && { echo "aborting."; rm -f "$KEEP" "$DROP"; exit 1; }

if [ "$DELETE" != 1 ]; then
    echo
    echo "DRY RUN. Re-run with --delete to actually remove."
    rm -f "$KEEP" "$DROP"; exit 0
fi

echo
read -r -p "delete ${#PLAN[@]} paths? type yes: " ans
[ "$ans" = yes ] || { echo "aborted."; rm -f "$KEEP" "$DROP"; exit 1; }
rm -rf "${PLAN[@]}"
echo "done."
df -h . | tail -1 | awk '{printf "disk now: %s used, %s available\n", $3, $4}'
rm -f "$KEEP" "$DROP"
