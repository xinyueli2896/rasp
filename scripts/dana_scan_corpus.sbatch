#!/bin/bash
# Scan the LA MIDI corpus into the two STAGES every version is built from, in
# one CPU job: download, scan, finalise.
#
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env sbatch scripts/dana_scan_corpus.sbatch
#
# The scan is version-agnostic, which is why this job is not named after one.
# It writes stage_bass and stage_chordal, and from those:
#
#   v4   = stage_bass,    roots only, bass line alone
#   v5   = stage_chordal, roots only, bass plus the chord tracks
#   v6   = stage_chordal  PLUS the chord-quality filter, which is a separate
#          --from_stage step once census_v6_quality has said which QUAL
#          patterns the corpus actually plays
#
# So going "straight to v6" still starts here: the stages are the expensive
# artifact, every version after them is a seconds-long rebuild, and the
# datasets this job writes (v4_bass, v5_chordal) are the roots-only ones.
#
# WORK is the only thing you must set -- point it at the new disk. Everything
# the job makes lives under it, and nothing is written to the repo.
#
# Submission-time overrides work as usual, e.g. a bigger slice:
#   sbatch --cpus-per-task=128 --time=24:00:00 scripts/dana_scan_corpus.sbatch
#
# RESUMABLE. If the job hits its time limit, submit it again, unchanged: the
# download resumes (huggingface_hub keeps the partial blob) and the scan
# resumes from its stage, which logs every member it has already read. Two
# runs give the same datasets as one uninterrupted run; that is checked in the
# repo against a zip scanned as 300 + 300 members.
#
# ---------------------------------------------------------------------------
# Why the steps are in this order
#
# v5 keeps the bass plus the tracks playing chords under it, and drops a file
# that has no chordal track ('no_chordal_track'). So v5's spans are a SUBSET
# of v4's. Span selection itself looks only at the bass line, which means the
# expensive pass -- every member of a 404,714-file corpus -- is the BASS one,
# and v5 then reads only the files that pass, via --only_files_from. Done the
# other way round, deriving v4 from a v5 stage would silently lose every
# bass-only song. Both datasets for roughly the price of one, on identical
# spans, so the v4 and v5 tables compare directly.
#
# The scans stage spans down to --min_bars 8, which is what the protocol uses.
# The stage keeps the span tensors, so changing the floor UPWARD later is a
# seconds-long --from_stage run, while lowering it would mean scanning the
# corpus again -- the part that costs hours. So stage low, whatever you build.
#
# On DANA's Lustre guidance about many small files: the corpus stays a single
# zip and is never extracted. UglyMIDI takes a file object, so members are
# parsed in memory. The job's own writes are ~200 batched stage files, not
# 404,714 of them.
# ---------------------------------------------------------------------------

#
# PARTITION AND QOS. DANA's docs give --partition=cscc-cpu-p with
# --qos=cscc-cpu-qos, and that is what this job asked for until a submission
# came back "Invalid qos specification": the docs state policy, while what a
# given account may request lives in its SLURM association. So no QoS is
# hardcoded here -- submitting without one uses your association's default,
# which is usually right. Add it on the command line once you know the name:
#
#   sacctmgr show assoc where user=$USER format=Account,Partition,QOS,DefQOS -p
#   sbatch --qos=<name> [-A <account>] ...
#
# Command-line flags override the directives below, so the partition can be
# redirected the same way.

#SBATCH --job-name=scan
#SBATCH --output=scan-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=12:00:00

set -euo pipefail

WORK="${WORK:?set WORK to a directory on the new disk, e.g. /l/users/$USER/v5}"
CONDA_ENV="${CONDA_ENV:-rasp}"
BRANCH="${BRANCH:-claude/modulo-sequence-generator-gzR8i}"
SCAN_MIN_BARS="${SCAN_MIN_BARS:-8}"     # what gets staged
# What the datasets are built at. 8 bars = 128 subbeats, which is exactly what
# this experiment's protocol uses: run_v4_stems.sh sets TRAIN_LENGTH=128 and
# evaluates with a 16-subbeat prompt and 112 generated, one bar plus seven.
# This default was 24 for a while, carried over from v1-v3's 384-subbeat
# window, and it cost 95% of the data: on the LA corpus the chordal pass gives
# 4032 spans at 8 bars against 184 at 24, and 288 held-out-key spans against
# 13. Raise it only for a protocol that really needs longer songs.
MIN_BARS="${MIN_BARS:-8}"
BATCH="${BATCH:-2000}"                  # members per dispatch round
MAX_FILES="${MAX_FILES:-0}"             # >0 for a probe run
# Extra flags for both scans, e.g. EXTRA="--chord_frac 0.15" to loosen what
# counts as a chord track, or "--cover_min 0.6" to accept a sparser bass.
EXTRA="${EXTRA:-}"

ZIP_NAME=Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA.zip
ZIP="$WORK/LAMD/$ZIP_NAME"
# The repo is the checkout you submitted from, not a copy the job makes: WORK
# may sit INSIDE it (the repo gitignores /v6/), and cloning into WORK would
# then nest the repo in itself. SLURM_SUBMIT_DIR is where sbatch was run.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
UPSTREAM="${UPSTREAM:-$WORK/mald-upstream}"
# Off by default: this is your working checkout, and the job has no business
# moving it under you. Pull before submitting, or pass PULL=1.
PULL="${PULL:-0}"

mkdir -p "$WORK"/{LAMD,data,stage_bass,stage_chordal,tmp,logs}

# One worker per core, and one thread per worker. Without this each joblib
# worker lets torch/BLAS open as many threads as there are cores, and 64
# workers times 64 threads thrashes the node for no extra throughput.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER="$WORK/tmp"
# Keep the download cache off the home quota; it holds the partial blob that
# makes a resumed download cheap.
export HF_HOME="$WORK/hf_home"
JOBS="${SLURM_CPUS_PER_TASK:-8}"

echo "host      $(hostname)"
echo "job       ${SLURM_JOB_ID:-interactive}"
echo "work      $WORK"
echo "workers   $JOBS"
echo "started   $(date -Is)"

# sbatch exports the submitting environment, so conda is usually already on
# PATH from your login shell. Where it comes from a module instead, name it:
#   MODULES="miniconda/3" WORK=... sbatch scripts/dana_scan_corpus.sbatch
# The env's interpreter is named directly rather than activated. `conda
# activate` is a shell function that conda init writes into your .bashrc, and a
# batch job never sources that, so activating inside a job needs a conda on
# PATH plus the shell hook -- and whether it is there depends on which shell
# you submitted from. Naming the interpreter needs none of it.
PY="${PY:-$CONDA_ENV/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "no interpreter at $PY." >&2
  echo "CONDA_ENV should be the env PREFIX (a path), e.g. \$PWD/v6/env." >&2
  echo "For an env known only by name, pass PY=/path/to/that/env/bin/python." >&2
  exit 1
fi
echo "python    $PY"
"$PY" -c "import torch, pretty_midi, joblib, huggingface_hub" \
  || { echo "environment $CONDA_ENV is missing deps; run scripts/dana_setup_env.sh" >&2; exit 1; }

# --- 1. code ---------------------------------------------------------------
if [[ ! -f "$REPO/midi_adapter/prepare_v4_data.py" ]]; then
  echo "REPO=$REPO is not a checkout of this repository." >&2
  echo "Submit from the repository root, or pass REPO=/path/to/rasp." >&2
  exit 1
fi
echo "repo      $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
if [[ "$PULL" == 1 ]]; then
  git -C "$REPO" fetch origin "$BRANCH" && git -C "$REPO" checkout "$BRANCH" \
    && git -C "$REPO" pull origin "$BRANCH"
fi
# The upstream preprocessing: prepare_v4_data imports UglyMIDI and
# preprocess_midi from it rather than reimplementing them, so the CP tensors
# match what the frozen v0.42 host was pretrained on.
[[ -d "$UPSTREAM/.git" ]] \
  || git clone --depth 1 https://github.com/instr3/MIDI-AutoLabel-Dataset "$UPSTREAM"

# --- 2. corpus -------------------------------------------------------------
if [[ -s "$ZIP" ]]; then
  echo "corpus already present: $ZIP ($(du -h "$ZIP" | cut -f1))"
else
  echo "downloading the LA MIDI corpus ..."
  "$PY" - <<PYEOF
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='projectlosangeles/Los-Angeles-MIDI-Dataset',
                    filename='$ZIP_NAME',
                    repo_type='dataset', local_dir='$WORK/LAMD')
print('downloaded', p)
PYEOF
fi
df -h "$WORK" | tail -1

cd "$REPO"
common=(--midi_zip "$ZIP" --upstream "$UPSTREAM" --jobs "$JOBS" --batch "$BATCH")
[[ "$MAX_FILES" != 0 ]] && common+=(--max_files "$MAX_FILES")
# shellcheck disable=SC2206  -- word splitting is what EXTRA is for
[[ -n "$EXTRA" ]] && common+=($EXTRA)

# prepare_v4_data exits non-zero when a pass finds no spans at all, which is
# a legitimate outcome for one track mode and not a reason to throw away the
# other. Each pass runs under `if` so `set -e` does not end the job on it.

# --- 3. the expensive pass: bass, every member -----------------------------
echo
echo "=== scan: bass, min_bars $SCAN_MIN_BARS ==="
if "$PY" -m midi_adapter.prepare_v4_data "${common[@]}" \
     --tracks bass --stage_dir "$WORK/stage_bass" \
     --min_bars "$SCAN_MIN_BARS" \
     --out_prefix "$WORK/data/v4_bass_min${SCAN_MIN_BARS}" \
     2>&1 | tee -a "$WORK/logs/scan_bass.log"; then
  echo "bass pass done"
else
  echo "bass pass found no spans -- see $WORK/logs/scan_bass.log for which" >&2
  echo "reject or bar reason dominated. Nothing downstream can run." >&2
  exit 1
fi

# --- 4. the cheap pass: chordal, only the files that yielded spans ---------
echo
echo "=== scan: chordal (v5), only the files behind the bass spans ==="
if "$PY" -m midi_adapter.prepare_v4_data "${common[@]}" \
     --tracks chordal --only_files_from "$WORK/stage_bass" \
     --stage_dir "$WORK/stage_chordal" \
     --min_bars "$SCAN_MIN_BARS" \
     --out_prefix "$WORK/data/v5_chordal_min${SCAN_MIN_BARS}" \
     2>&1 | tee -a "$WORK/logs/scan_chordal.log"; then
  echo "chordal pass done"
else
  echo "chordal pass found no spans -- 'no_chordal_track' in the log means" >&2
  echo "these songs carry no track that both belongs to a chordal GM family" >&2
  echo "and plays 2+ notes on --chord_frac of its steps. v4 is unaffected." >&2
fi

# --- 5. the datasets, at the training window's floor -----------------------
for pair in "bass:v4_bass" "chordal:v5_chordal"; do
  stage="stage_${pair%%:*}"; prefix="${pair##*:}"
  if ! compgen -G "$WORK/$stage/spans/*.pt" > /dev/null; then
    echo; echo "=== $prefix: stage is empty, nothing to finalise ==="
    continue
  fi
  echo
  echo "=== finalise $prefix at min_bars $MIN_BARS ==="
  "$PY" -m midi_adapter.prepare_v4_data \
    --from_stage "$WORK/$stage" --upstream "$UPSTREAM" \
    --tracks "${pair%%:*}" --min_bars "$MIN_BARS" \
    --out_prefix "$WORK/data/$prefix" \
    2>&1 | tee -a "$WORK/logs/finalise_$prefix.log"
done

echo
echo "finished  $(date -Is)"
echo "datasets  $WORK/data/{v4_bass,v5_chordal}_{finetune,unseen,all}.pt"
echo "manifests $WORK/data/{v4_bass,v5_chordal}.manifest.json"
echo "stages    $WORK/{stage_bass,stage_chordal}   (keep these: rebuilding a"
echo "          dataset at another floor or with --balance_keys needs no rescan)"
echo
echo "Check in the logs, before training on any of it:"
echo "  * the three invariants read 1.000 (and for v5, slot 0 is the bass root)"
echo "  * the per-key table -- if the held-out keys F# and G# are thin, rerun"
echo "    the finalise step with --balance_keys N, which takes seconds:"
echo "    "$PY" -m midi_adapter.prepare_v4_data --from_stage $WORK/stage_chordal \\"
echo "        --upstream $UPSTREAM --tracks chordal --min_bars $MIN_BARS \\"
echo "        --balance_keys N --out_prefix $WORK/data/v5_chordal"
