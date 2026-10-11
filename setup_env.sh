#!/usr/bin/env bash
# =============================================================================
#  Build the environment for this repo, then PROVE it works.
#
#  SLURM-AWARE. verify_env.py builds the 112M CP transformer and runs a
#  forward pass, which does not belong on a login node. Where `sbatch` exists
#  and we are not already inside a job, this installs and then SUBMITS the
#  verification instead of running it.
#
#  The proof matters more than the install. Several ways this repo can be
#  half-broken do not surface until a training run is hours in:
#
#    * `wandb` missing     — the trainers import WandbLogger unconditionally,
#                            so it fails at import, not at log time
#    * wrong `lightning`   — every module does `import pytorch_lightning as L`;
#                            the newer `lightning` package does NOT provide it
#    * transformers bump   — cp_transformer.py imports RoFormerEncoder from a
#                            non-public path, the most likely thing to break
#    * torch without CUDA  — installs cleanly, then trains at 1/50th speed
#
#  Usage
#    bash setup_env.sh                 # install, then verify (job if Slurm)
#    bash setup_env.sh --verify        # verification step only
#    bash setup_env.sh --compile       # also install tracr/jax (rule compiling)
#    bash setup_env.sh --venv          # python -m venv instead of conda
#    bash setup_env.sh --local         # run verification HERE, not via sbatch
#                                      #   (only inside `srun --pty`; MBZUAI
#                                      #    policy says not to use salloc as
#                                      #    the standard way to run work)
#    bash setup_env.sh --no-verify     # install only
#
#    bash setup_env.sh                           # DANA: partition `long`
#    SB_PARTITION=cscc-gpu-p bash setup_env.sh   # if `long` is saturated
#    ENV_NAME=rasp SB_ACCOUNT=myproj SB_GRES=gpu:1 bash setup_env.sh
#
#  See slurm/README.md for offline wandb, multi-GPU srun, and preemption.
# =============================================================================

set -uo pipefail

ENV_NAME="${ENV_NAME:-rasp}"          # the pipeline scripts hardcode `rasp`
PY_VER="${PY_VER:-3.11}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DO_INSTALL=1; DO_VERIFY=1; DO_COMPILE=0; USE_VENV=0; FORCE_LOCAL=0
for a in "$@"; do
    case "$a" in
        --verify)    DO_INSTALL=0 ;;
        --no-verify) DO_VERIFY=0 ;;
        --compile)   DO_COMPILE=1 ;;
        --venv)      USE_VENV=1 ;;
        --local)     FORCE_LOCAL=1 ;;
        -h|--help)   sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 1 ;;
    esac
done

say() { echo -e "\n── $* ──"; }

# On a Slurm login node when sbatch exists and we are not inside a job.
ON_SLURM_LOGIN=0
if command -v sbatch >/dev/null 2>&1 && [ -z "${SLURM_JOB_ID:-}" ]; then
    ON_SLURM_LOGIN=1
fi

# ── install ──────────────────────────────────────────────────────────────
# pip is IO-bound, not compute, so it is normally fine on a login node. If
# your cluster forbids even that, run this whole script inside an allocation.
if [ "$DO_INSTALL" = 1 ]; then
    if [ "$USE_VENV" = 1 ]; then
        say "venv at $REPO/.venv (python $PY_VER)"
        "python${PY_VER}" -m venv "$REPO/.venv" || python3 -m venv "$REPO/.venv" || exit 1
        # shellcheck disable=SC1091
        source "$REPO/.venv/bin/activate"
        RUN=""
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "conda not found. Load your cluster's module (e.g. " >&2
            echo "  module load anaconda3) or re-run with --venv." >&2
            exit 1
        fi
        if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
            say "conda env '$ENV_NAME' exists, reusing it"
        else
            say "creating conda env '$ENV_NAME' (python $PY_VER)"
            conda create -y -n "$ENV_NAME" "python=$PY_VER" || exit 1
        fi
        RUN="conda run -n $ENV_NAME --live-stream --no-capture-output"
    fi

    say "installing core requirements"
    # Plain PyPI gives the CUDA build of torch. Do NOT point at the CPU index
    # unless you mean it: it installs cleanly and then trains far slower. If
    # this cluster needs a specific CUDA build, install torch by hand first
    # and then run `bash setup_env.sh --verify`.
    $RUN python -m pip install --upgrade pip || exit 1
    $RUN python -m pip install -r "$REPO/requirements.txt" || exit 1

    if [ "$DO_COMPILE" = 1 ]; then
        say "installing the rule-COMPILING extras (tracr + jax)"
        $RUN python -m pip install -r "$REPO/requirements-compile.txt" || exit 1
    fi
else
    RUN=""
fi

[ "$DO_VERIFY" = 1 ] || { echo; echo "Installed. Verify with: bash setup_env.sh --verify"; exit 0; }

# ── verify ───────────────────────────────────────────────────────────────
cd "$REPO" || exit 1

if [ "$ON_SLURM_LOGIN" = 1 ] && [ "$FORCE_LOCAL" = 0 ]; then
    say "Slurm detected — submitting verification as a job"
    echo "   verify_env.py builds the 112M CP transformer, so it does not run here."
    mkdir -p slurm_logs
    SB_ARGS=()
    # `long` is DANA's default GPU partition (39 nodes); cscc-gpu-p (16) is
    # the alternative. verify.sbatch already asks for `long`, so this only
    # has an effect when overridden.
    [ -n "${SB_PARTITION:-}" ] && SB_ARGS+=(--partition="$SB_PARTITION")
    [ -n "${SB_ACCOUNT:-}" ]   && SB_ARGS+=(--account="$SB_ACCOUNT")
    [ -n "${SB_GRES:-}" ]      && SB_ARGS+=(--gres="$SB_GRES")
    # verify.sbatch already sets --qos=gpu-debug-qos, which is what DANA
    # policy requires for environment/GPU verification. SB_QOS overrides it
    # only if your site names it differently.
    [ -n "${SB_QOS:-}" ]       && SB_ARGS+=(--qos="$SB_QOS")

    out=$(ENV_NAME="$ENV_NAME" sbatch "${SB_ARGS[@]}" slurm/verify.sbatch 2>&1)
    echo "   $out"
    jid=$(echo "$out" | grep -oE '[0-9]+$' || true)
    echo
    if [ -n "$jid" ]; then
        echo "Submitted. When it finishes:"
        echo "   cat slurm_logs/verify-$jid.out        # expect: environment OK"
    else
        echo "sbatch did not return a job id — check the resource flags above."
        echo "Set SB_PARTITION / SB_ACCOUNT / SB_GRES for your cluster, or edit"
        echo "the #SBATCH lines in slurm/verify.sbatch."
        exit 1
    fi
    echo
    echo "Then:  conda activate $ENV_NAME"
    echo "See slurm/README.md for offline wandb, multi-GPU srun, preemption."
    exit 0
fi

say "verifying here"
if [ "$ON_SLURM_LOGIN" = 1 ]; then
    echo "   (--local on a Slurm login node — only do this inside an"
    echo "    interactive job: srun --qos=gpu-debug-qos --gres=gpu:1 --pty bash"
    echo "    Running it on the login node itself can get the process killed"
    echo "    and your SSH session closed.)"
fi
$RUN python "$REPO/verify_env.py"
rc=$?

echo
if [ "$rc" = 0 ]; then
    if [ "$USE_VENV" = 1 ]; then
        echo "Ready.  source $REPO/.venv/bin/activate"
    else
        echo "Ready.  conda activate $ENV_NAME"
    fi
else
    echo "Environment is NOT ready — see the failures above."
fi
exit $rc
