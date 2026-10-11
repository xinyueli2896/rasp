#!/usr/bin/env bash
# =============================================================================
#  Build the environment for this repo on a fresh machine, then PROVE it works.
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
#  So --verify runs a real forward pass of the 112M CP transformer and the
#  rule-model test suite, rather than just importing things.
#
#  Usage
#    bash setup_env.sh                 # create env + install + verify
#    bash setup_env.sh --verify        # verify only, in the CURRENT env
#    bash setup_env.sh --compile       # also install tracr/jax (rule compiling)
#    bash setup_env.sh --venv          # use python -m venv instead of conda
#    ENV_NAME=myenv bash setup_env.sh
# =============================================================================

set -uo pipefail

ENV_NAME="${ENV_NAME:-rasp}"          # the pipeline scripts hardcode `rasp`
PY_VER="${PY_VER:-3.11}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DO_INSTALL=1; DO_COMPILE=0; USE_VENV=0
for a in "$@"; do
    case "$a" in
        --verify)  DO_INSTALL=0 ;;
        --compile) DO_COMPILE=1 ;;
        --venv)    USE_VENV=1 ;;
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 1 ;;
    esac
done

say() { echo -e "\n── $* ──"; }

# ── install ──────────────────────────────────────────────────────────────
if [ "$DO_INSTALL" = 1 ]; then
    if [ "$USE_VENV" = 1 ]; then
        say "venv at $REPO/.venv (python $PY_VER)"
        python${PY_VER} -m venv "$REPO/.venv" || python3 -m venv "$REPO/.venv" || exit 1
        # shellcheck disable=SC1091
        source "$REPO/.venv/bin/activate"
        RUN=""
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "conda not found. Either load/activate it, or re-run with --venv." >&2
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
    # unless you mean it: it installs cleanly and then trains ~50x slower.
    $RUN python -m pip install --upgrade pip || exit 1
    $RUN python -m pip install -r "$REPO/requirements.txt" || exit 1

    if [ "$DO_COMPILE" = 1 ]; then
        say "installing the rule-COMPILING extras (tracr + jax)"
        $RUN python -m pip install -r "$REPO/requirements-compile.txt" || exit 1
    fi
else
    RUN=""
fi

# ── verify ───────────────────────────────────────────────────────────────
say "verifying"
cd "$REPO" || exit 1
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
