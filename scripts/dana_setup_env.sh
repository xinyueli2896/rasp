#!/usr/bin/env bash
# Create the conda environment for this repo on DANA (the cluster the docs
# still name cscc-* in SLURM identifiers). Run it ON THE LOGIN NODE: creating
# an environment is "preparing a workload", which the login nodes are for, and
# it stays inside their 8-core / 16 GB slice. Nothing here is compute.
#
#   bash scripts/dana_setup_env.sh                 # env "rasp", prep deps only
#   WITH_TRAINING=1 bash scripts/dana_setup_env.sh # also the training stack
#   ENV_NAME=rasp-v5 ENV_PREFIX=/l/users/$USER/envs/rasp-v5 bash scripts/...
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
echo "done. The preparation job activates it with:"
if [[ -n "$ENV_PREFIX" ]]; then
  echo "  CONDA_ENV=$ENV_PREFIX sbatch scripts/dana_v5_prepare.sbatch"
else
  echo "  CONDA_ENV=$activate sbatch scripts/dana_v5_prepare.sbatch"
fi
