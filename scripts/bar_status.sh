#!/usr/bin/env bash
# Where is the bar experiment up to?
#
# Reads only what the driver already writes -- checkpoint filenames carry the
# monitored unseen_acc, so the best score so far needs no log parsing and no
# wandb. Safe to run at any time; it touches nothing.
#
#   CKPT_DIR / LOG_DIR  same values passed to run_bar_experiment.sh
#   DRIVER              the nohup log (default run_<LOG_DIR>_driver.log)
#
# Usage:  CKPT_DIR=checkpoints_bar4 LOG_DIR=logs_bar4 ./scripts/bar_status.sh
#         watch -n 60 'CKPT_DIR=checkpoints_bar4 LOG_DIR=logs_bar4 ./scripts/bar_status.sh'

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CKPT_DIR="${CKPT_DIR:-checkpoints_bar}"
LOG_DIR="${LOG_DIR:-logs_bar}"
DRIVER="${DRIVER:-run_${LOG_DIR}_driver.log}"

echo "ckpt=$CKPT_DIR  log=$LOG_DIR  $(date '+%F %T')"
echo

# --- what is actually running ---------------------------------------------
n_proc=$(pgrep -fc 'midi_adapter.(train_cp_yinyang|finetune_cp_bass)' 2>/dev/null)
echo "training processes: $n_proc"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>/dev/null \
    | sed 's/^/  gpu /'
fi
echo

# --- per-run progress ------------------------------------------------------
# The monitor is unseen_acc with mode=max, so the highest value in the
# filenames is the current best and the count shows how many improvements
# there have been.
printf '%-30s %5s %10s %7s  %s\n' RUN CKPTS BEST_UNSEEN EPOCH LAST_WRITE
printf '%-30s %5s %10s %7s  %s\n' '------------------------------' '-----' '----------' '-------' '----------------'
shopt -s nullglob
found=0
for d in "$CKPT_DIR"/*/; do
  n="$(basename "$d")"
  cks=("$d"*.ckpt)
  (( ${#cks[@]} )) || continue
  found=1
  # NB: nullglob is on, so a glob that matches nothing disappears entirely --
  # `ls -1 <vanished glob>` would list the CURRENT DIRECTORY instead. Collect
  # into an array and test it rather than letting ls see an empty argument list.
  accs=("$d"*unseen_acc=*.ckpt)
  if (( ${#accs[@]} )); then
    best="$(printf '%s\n' "${accs[@]}" \
            | sed 's/.*unseen_acc=//; s/\.ckpt$//' | sort -g | tail -1)"
    ep="$(printf '%s\n' "${accs[@]}" | grep -F "unseen_acc=$best.ckpt" \
          | head -1 | sed 's/.*epoch=//; s/\..*//')"
  else
    # the finetune baseline is selected on val_loss, not unseen_acc
    vls=("$d"*val_loss=*.ckpt)
    if (( ${#vls[@]} )); then
      best="vl:$(printf '%s\n' "${vls[@]}" \
                 | sed 's/.*val_loss=//; s/\.ckpt$//' | sort -g | head -1)"
    else
      best=''
    fi
    ep='-'
  fi
  newest="$(printf '%s\n' "${cks[@]}" | xargs -r ls -1t 2>/dev/null | head -1)"
  age="$(date -r "$newest" '+%m-%d %H:%M' 2>/dev/null || echo '?')"
  printf '%-30s %5d %10s %7s  %s\n' "$n" "${#cks[@]}" "${best:--}" "${ep:--}" "$age"
done
(( found )) || echo "  (no checkpoints yet)"
echo

# --- tables written so far -------------------------------------------------
tabs=("$LOG_DIR"/table_*.log)
if (( ${#tabs[@]} )); then
  echo "tables:"
  for t in "${tabs[@]}"; do
    # The Overall line is the one-line summary of a finished table.
    ok=$(grep -c '^Overall' "$t" 2>/dev/null)
    printf '  %-34s %s\n' "$(basename "$t")" \
           "$([[ "$ok" != 0 ]] && echo complete || echo 'in progress / failed')"
  done
  echo
fi

# --- anything gone wrong ---------------------------------------------------
if [[ -f "$DRIVER" ]]; then
  echo "driver ($DRIVER), last 8 lines:"
  tail -8 "$DRIVER" | sed 's/^/  /'
  fails="$(grep -c 'FAILED' "$DRIVER" 2>/dev/null)"
  [[ "$fails" != 0 ]] && echo "  !! $fails FAILED line(s) -- grep FAILED $DRIVER"
else
  echo "driver log not found: $DRIVER  (set DRIVER=...)"
fi

errs="$(grep -l -iE 'Traceback|CUDA out of memory|SystemExit' "$LOG_DIR"/*.log 2>/dev/null | head -5)"
if [[ -n "$errs" ]]; then
  echo
  echo "logs containing errors:"
  echo "$errs" | sed 's/^/  /'
fi
true
