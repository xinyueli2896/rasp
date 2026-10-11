# Sourced by every job script. Edit THIS file once for a new cluster.
#
# RASP_REPO / D / BASE are read by midi_adapter/run_*.sh as overridable
# defaults, so exporting them here redirects every run script at once.

# Repo and data roots.
export RASP_REPO="${RASP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export D="${D:-$RASP_REPO/../data/pop909_ivvi_w1}"
export BASE="${BASE:-checkpoints/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt}"

# Conda env created by setup_env.sh.
ENV_NAME="${ENV_NAME:-rasp}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" && conda activate "$ENV_NAME"
elif [ -f "$RASP_REPO/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$RASP_REPO/.venv/bin/activate"
fi

# Compute nodes may have no outbound network, and an online wandb run can
# hang on connection retry rather than fail. Sync afterwards from the login
# node: `wandb sync wandb/offline-run-*`.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "node    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}   partition: ${SLURM_JOB_PARTITION:-none}"
echo "gpus    : ${CUDA_VISIBLE_DEVICES:-none}"   # read-only; never set this
echo "repo    : $RASP_REPO"
echo "data    : $D"
echo
