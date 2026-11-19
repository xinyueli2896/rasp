#!/bin/bash
# Rebuild the datasets from the stages, and census their chord qualities.
# Everything downstream of the corpus scan, as one batch job.
#
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env sbatch \
#       -o $PWD/v6/logs/finalise-%j.out scripts/dana_finalise.sbatch
#
# The scan (scripts/dana_scan_corpus.sbatch) is the expensive part and runs
# once: it reads 404,714 MIDI members and writes stage_bass and stage_chordal.
# This job touches no MIDI at all -- it reads those stages' span tensors -- so
# it finishes in seconds to minutes and can be re-run freely. Every variation
# of the data is a run of this job:
#
#   MIN_BARS=12 ...        a higher floor on span length
#   BALANCE_KEYS=200 ...   transpose spans so each key has at least 200
#   CENSUS=0 ...           skip the quality census
#
# MIN_BARS defaults to 8 because that is what the protocol uses: 8 bars = 128
# subbeats, run_v4_stems.sh's TRAIN_LENGTH, evaluated as a 16-subbeat prompt
# plus 112 generated. Only raise it for a protocol that needs longer songs --
# on the LA corpus a floor of 24 costs 95% of the spans and leaves 13 in the
# held-out keys.
#
# PARTITION AND QOS. No QoS is requested: cscc-cpu-p carries cscc-cpu-qos as
# its PARTITION QoS, applied automatically, while --qos asks for a JOB QoS
# from your SLURM association, which does not list it ("Invalid qos
# specification"). Add one on the command line if that ever changes.
#
# MODULES="..." loads modules first, for a conda that comes from one: a batch
# job does not read your .bashrc.

#SBATCH --job-name=finalise
#SBATCH --output=finalise-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=01:00:00

set -euo pipefail

# An apostrophe inside ${var:?word} opens a quote even within double
# quotes, which is why this message has none.
WORK="${WORK:?set WORK to the scan work directory, e.g. \$PWD/v6}"
CONDA_ENV="${CONDA_ENV:-rasp}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
UPSTREAM="${UPSTREAM:-$WORK/mald-upstream}"
MIN_BARS="${MIN_BARS:-8}"
BALANCE_KEYS="${BALANCE_KEYS:-0}"
CENSUS="${CENSUS:-1}"

mkdir -p "$WORK/logs"

if [[ -n "${MODULES:-}" ]]; then
  # shellcheck disable=SC1091
  [[ -s /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
  # shellcheck disable=SC2086
  module load $MODULES
fi
if ! command -v conda >/dev/null; then
  echo "conda not on PATH. A batch job does not read your .bashrc, so if" >&2
  echo "conda comes from a module, resubmit with MODULES=\"<that module>\"." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [[ ! -f "$REPO/midi_adapter/prepare_v4_data.py" ]]; then
  echo "REPO=$REPO is not a checkout of this repository." >&2
  echo "Submit from the repository root, or pass REPO=/path/to/rasp." >&2
  exit 1
fi
cd "$REPO"

echo "host      $(hostname)"
echo "repo      $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "work      $WORK"
echo "min_bars  $MIN_BARS"
echo "balance   $BALANCE_KEYS"
echo "started   $(date -Is)"

# --- the datasets ----------------------------------------------------------
for pair in "bass:v4_bass" "chordal:v5_chordal"; do
  stage="stage_${pair%%:*}"; prefix="${pair##*:}"
  if ! compgen -G "$WORK/$stage/spans/*.pt" > /dev/null; then
    echo; echo "=== $prefix: $stage holds no spans, skipping ==="
    continue
  fi
  echo
  echo "=== $prefix from $stage at min_bars $MIN_BARS ==="
  args=(--from_stage "$WORK/$stage" --upstream "$UPSTREAM"
        --tracks "${pair%%:*}" --min_bars "$MIN_BARS"
        --out_prefix "$WORK/data/$prefix")
  [[ "$BALANCE_KEYS" != 0 ]] && args+=(--balance_keys "$BALANCE_KEYS")
  python -m midi_adapter.prepare_v4_data "${args[@]}" \
    2>&1 | tee "$WORK/logs/finalise_${prefix}.log"
done

# --- the v6 quality census -------------------------------------------------
# Only the chordal stage can answer this: a bass line alone voices no third,
# so it carries no quality to read.
if [[ "$CENSUS" == 1 ]] && compgen -G "$WORK/stage_chordal/spans/*.pt" > /dev/null; then
  echo
  echo "=== v6 quality census over stage_chordal ==="
  python -m midi_adapter.census_v6_quality \
    --stage "$WORK/stage_chordal" --min_bars "$MIN_BARS" \
    --json_out "$WORK/logs/census_min${MIN_BARS}.json" \
    2>&1 | tee "$WORK/logs/census_min${MIN_BARS}.log"
fi

echo
echo "finished  $(date -Is)"
echo "datasets  $WORK/data/{v4_bass,v5_chordal}_{finetune,unseen,pretrain,all}.pt"
echo "census    $WORK/logs/census_min${MIN_BARS}.{log,json}"
echo
echo "Before training on any of it, the invariants above must read 1.000 --"
echo "and for the chordal set, 'downbeat slot 0 is the bass root' is the one"
echo "that counts, since the rule model and the probe both read slot 0."
