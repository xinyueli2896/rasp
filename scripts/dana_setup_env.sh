#!/usr/bin/env bash
# Create the conda environment for this repo on DANA (the cluster the docs
# still name cscc-* in SLURM identifiers).
#
# RUN IT UNDER srun, not on the login node. Creating an environment counts as
# "preparing a workload", but two steps here are not as light as that sounds:
# conda's dependency solve can hold a core for a minute or more, and
# `pip install torch` unpacks a couple of GB and byte-compiles thousands of
# files -- several minutes of sustained CPU, which is what the login nodes'
# watchdog is looking for. It would kill the process AND close the SSH session.
# Nothing is lost if that happens (this script is idempotent), but one line
# avoids it, and DANA's own docs list dependency validation as legitimate
# interactive use:
#
#   srun --partition=cscc-cpu-p --qos=cscc-cpu-qos --cpus-per-task=4 \
#        --mem=16G --time=01:00:00 \
#        env ENV_PREFIX=$PWD/v6/env bash scripts/dana_setup_env.sh
#
# Pass the script AS the srun command rather than taking --pty bash and typing
# it, so the allocation ends when the install does: idle allocations are
# against the rules too.
#
#   WITH_TRAINING=1 ...   also install the training stack
#   ENV_NAME=rasp-v6 ...  name it, instead of giving a path
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

set -euo pipefail

ENV_NAME="${ENV_NAME:-rasp}"
ENV_PREFIX="${ENV_PREFIX:-}"
PY_VERSION="${PY_VERSION:-3.11}"
WITH_TRAINING="${WITH_TRAINING:-0}"

if ! command -v conda >/dev/null; then
  echo "conda not on PATH. Load it the way you do for the mitosis project" >&2
  echo "(e.g. 'module load miniconda' or source its profile.d script)." >&2
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
echo "    sbatch -o \$PWD/v6/logs/scan-%j.out scripts/dana_v5_prepare.sbatch"
