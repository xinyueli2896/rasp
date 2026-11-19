#!/usr/bin/env bash
# Create the conda environment for this repo on DANA (the cluster the docs
# still name cscc-* in SLURM identifiers).
#
# SUBMIT IT, do not run it on the login node. It carries its own #SBATCH
# directives, so:
#
#   ENV_PREFIX=$PWD/v6/env sbatch -o $PWD/v6/logs/env-%j.out \
#       scripts/dana_setup_env.sh
#
# Creating an environment counts as "preparing a workload", but two steps here
# are not as light as that sounds: conda's dependency solve can hold a core for
# a minute or more, and `pip install torch` unpacks a couple of GB and
# byte-compiles thousands of files -- several minutes of sustained CPU, which
# is what the login nodes' watchdog looks for. It would kill the process AND
# close the SSH session with it. The script is idempotent so nothing is lost,
# but submitting it costs one word and also survives a dropped connection,
# which a multi-GB download over a flaky link will not reliably do.
#
# To chain the scan behind it and walk away:
#
#   JOB=$(ENV_PREFIX=$PWD/v6/env sbatch --parsable \
#           -o $PWD/v6/logs/env-%j.out scripts/dana_setup_env.sh)
#   WORK=$PWD/v6 CONDA_ENV=$PWD/v6/env sbatch --dependency=afterok:$JOB \
#       -o $PWD/v6/logs/scan-%j.out scripts/dana_scan_corpus.sbatch
#
# srun with the same flags works too when you want to watch it live, and plain
# `bash` still works wherever running it directly is appropriate -- the
# directives are only comments to a shell.
#
#   WITH_TRAINING=1 ...   also install the training stack
#   ENV_NAME=rasp-v6 ...  name it, instead of giving a path
#   MODULES="..."         modules to load first, if conda comes from one
#
# ENV_PREFIX puts the environment on a project filesystem instead of your home
# quota: a torch install is a couple of GB and home quotas are small. With it
# set, activate by path (conda activate "$ENV_PREFIX"), which the job script
# does for you.
#
# Two pins are deliberate:
#   transformers==4.52.4  cp_transformer.py reaches into
#       transformers.models.roformer.modeling_roformer and passes
#       sinusoidal_pos through RoFormerEncoder. That is internal API: it moves
#       between releases, and this is the version the upstream project
#       (instr3/MIDI-AutoLabel-Dataset) pins, so it is the version the v0.42
#       checkpoint's code was written against.
#   pretty_midi>=0.2.11   0.2.10 breaks under numpy 2.

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

#SBATCH --job-name=rasp-env
#SBATCH --output=rasp-env-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=01:00:00

set -euo pipefail

ENV_NAME="${ENV_NAME:-rasp}"
ENV_PREFIX="${ENV_PREFIX:-}"
PY_VERSION="${PY_VERSION:-3.11}"
WITH_TRAINING="${WITH_TRAINING:-0}"

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

if [[ -n "$ENV_PREFIX" ]]; then
  target=(-p "$ENV_PREFIX")
  activate="$ENV_PREFIX"
else
  target=(-n "$ENV_NAME")
  activate="$ENV_NAME"
fi

if conda env list | grep -qE "(^|[[:space:]])${activate//\//\\/}([[:space:]]|$)"; then
  echo "environment $activate already exists -- activating it and installing into it"
else
  conda create -y "${target[@]}" "python=$PY_VERSION"
fi
conda activate "$activate"

# What the v5 data preparation needs, and nothing more. torch is here because
# prepare_v4_data writes torch tensors, not because anything trains: the
# preparation job is CPU-only and asks for no GPU.
python -m pip install -U pip
python -m pip install \
  "numpy>=2" torch "pretty_midi>=0.2.11" mido six joblib huggingface_hub

if [[ "$WITH_TRAINING" == "1" ]]; then
  # The adapter stems and the tables. Only needed once you train on DANA's GPU
  # partitions; the preparation job never imports them.
  python -m pip install \
    "transformers==4.52.4" "pytorch_lightning>=2.5.3" "peft>=0.15.2" matplotlib
fi

echo
echo "import check:"
python - <<'PY'
import importlib
for m in ('numpy', 'torch', 'pretty_midi', 'mido', 'six', 'joblib',
          'huggingface_hub'):
    mod = importlib.import_module(m)
    print(f'  {m:16s} {getattr(mod, "__version__", "?")}')
for m in ('transformers', 'pytorch_lightning', 'peft', 'matplotlib'):
    try:
        mod = importlib.import_module(m)
        print(f'  {m:16s} {getattr(mod, "__version__", "?")}')
    except ImportError:
        print(f'  {m:16s} not installed (training stack; WITH_TRAINING=1)')
PY

echo
echo "done. Submit the preparation job FROM THE REPOSITORY ROOT, so it finds"
echo "the code through SLURM_SUBMIT_DIR:"
echo "  WORK=\$PWD/v6 CONDA_ENV=$activate \\"
echo "    sbatch -o \$PWD/v6/logs/scan-%j.out scripts/dana_scan_corpus.sbatch"
