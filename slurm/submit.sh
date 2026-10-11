#!/usr/bin/env bash
# =============================================================================
#  Submit any command in this repo as a DANA job, with the partition/QoS
#  pairing and the resource caps applied for you.
#
#  On DANA the partition and QoS are PAIRED, and mismatching them gets the
#  job rejected:
#
#      cscc-gpu-p + cscc-gpu-qos    standard GPU      <=  4 GPUs/account,
#                                                         2 node-jobs, 24 h
#      cscc-gpu-p + gpu-debug-qos   debug/interactive  (short work only)
#      long       + gpu-12          large/general GPU <= 12 GPUs, 8 jobs, 72 h
#      cscc-cpu-p + cscc-cpu-qos    CPU only          <= 512 cores, 2 jobs, 72 h
#
#  Usage
#    bash slurm/submit.sh -- python -m models.test_integer_rule_model
#    bash slurm/submit.sh --debug  -- python verify_env.py
#    bash slurm/submit.sh --cpu --cpus 32 --time 04:00:00 -- python -m ...
#    bash slurm/submit.sh --gpus 4 --time 48:00:00 -- python -m ...
#    bash slurm/submit.sh --dry    -- python -m ...      # print, do not submit
#
#  Options
#    --debug          cscc-gpu-p + gpu-debug-qos  (short tests; NOT training)
#    --standard       cscc-gpu-p + cscc-gpu-qos   (<= 4 GPUs, <= 24 h)
#    --cpu            cscc-cpu-p + cscc-cpu-qos   (no GPU)
#    (default)        long + gpu-12               (<= 12 GPUs, <= 72 h)
#    --gpus N         GPUs for this job (default 1)
#    --cpus N         CPU cores per task
#    --mem X          system RAM, e.g. 64G   (--mem is NOT GPU memory)
#    --time HH:MM:SS  wall clock
#    --name NAME      job name / log prefix
#    --dry            print the script and the sbatch line, submit nothing
#
#  Multi-GPU: with --gpus N > 1 this sets --ntasks-per-node=N and launches
#  under `srun`, because both trainers use devices=-1 and Lightning expects
#  one task per GPU. Bare `python` with N GPUs visible is the classic way to
#  get a hang or a single-process run.
#
#  Not used anywhere, deliberately: --exclusive (needs approval) and
#  --mail-type/--mail-user (unavailable on DANA).
# =============================================================================

set -uo pipefail

MODE=long; GPUS=1; CPUS=""; MEM=""; TIME=""; NAME=""; DRY=0
CMD=()
while [ $# -gt 0 ]; do
    case "$1" in
        --debug)    MODE=debug ;;
        --standard) MODE=standard ;;
        --cpu)      MODE=cpu ;;
        --gpus)     GPUS="$2"; shift ;;
        --cpus)     CPUS="$2"; shift ;;
        --mem)      MEM="$2";  shift ;;
        --time)     TIME="$2"; shift ;;
        --name)     NAME="$2"; shift ;;
        --dry)      DRY=1 ;;
        --)         shift; CMD=("$@"); break ;;
        -h|--help)  sed -n '2,42p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1  (did you forget '--' before the command?)" >&2
           exit 1 ;;
    esac
    shift
done

[ ${#CMD[@]} -gt 0 ] || { echo "nothing to run; put the command after --" >&2; exit 1; }

case "$MODE" in
    debug)    PART=cscc-gpu-p; QOS=gpu-debug-qos; MAXH=3;  MAXGPU=4 ;;
    standard) PART=cscc-gpu-p; QOS=cscc-gpu-qos;  MAXH=24; MAXGPU=4 ;;
    long)     PART=long;       QOS=gpu-12;        MAXH=72; MAXGPU=12 ;;
    cpu)      PART=cscc-cpu-p; QOS=cscc-cpu-qos;  MAXH=72; MAXGPU=0 ;;
esac

# Defaults follow the shapes in the DANA examples: 8 cpus / 40-64 G for one
# GPU, 16 cpus / 230 G per node for four.
[ "$MODE" = cpu ] && GPUS=0
if [ -z "$CPUS" ]; then
    if   [ "$MODE" = cpu ];  then CPUS=32
    elif [ "$GPUS" -ge 4 ];  then CPUS=16
    else                          CPUS=8
    fi
fi
if [ -z "$MEM" ]; then
    if   [ "$MODE" = cpu ]; then MEM=64G
    elif [ "$GPUS" -ge 4 ]; then MEM=230G
    else                         MEM=64G
    fi
fi
[ -n "$TIME" ] || TIME=$( [ "$MODE" = debug ] && echo 01:00:00 || echo 12:00:00 )
[ -n "$NAME" ] || NAME="rasp-$(basename "${CMD[-1]}" | tr -cd 'A-Za-z0-9_.-' | cut -c1-24)"

# ── refuse what the scheduler would reject anyway ────────────────────────
die() { echo "REFUSING: $*" >&2; exit 1; }
if [ "$MODE" != cpu ]; then
    [ "$GPUS" -ge 1 ] || die "$PART has a minimum request of 1 GPU"
    [ "$GPUS" -le "$MAXGPU" ] || \
        die "$GPUS GPUs exceeds the $MAXGPU-GPU cap on $PART/$QOS"
else
    [ "$CPUS" -le 512 ] || die "$CPUS cores exceeds the 512-core cap on $PART"
fi
hrs=$(( 10#$(echo "$TIME" | cut -d: -f1) ))
[ "$hrs" -le "$MAXH" ] || die "$TIME exceeds the ${MAXH}h limit on $PART/$QOS"
if [ "$MODE" = debug ]; then
    echo "NOTE: gpu-debug-qos is for short tests, debugging and GPU" >&2
    echo "      verification. Production training must use --standard or" >&2
    echo "      the default (long + gpu-12)." >&2
fi

mkdir -p slurm_logs
SCRIPT=$(mktemp "slurm_logs/${NAME}.XXXXXX.sbatch")
{
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=$NAME"
    echo "#SBATCH --output=slurm_logs/${NAME}-%j.out"
    echo "#SBATCH --error=slurm_logs/${NAME}-%j.out"
    echo "#SBATCH --partition=$PART"
    echo "#SBATCH --qos=$QOS"
    echo "#SBATCH --time=$TIME"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --cpus-per-task=$CPUS"
    echo "#SBATCH --mem=$MEM"
    [ "$GPUS" -gt 0 ] && echo "#SBATCH --gres=gpu:$GPUS"
    if [ "$GPUS" -gt 1 ]; then
        echo "#SBATCH --ntasks-per-node=$GPUS"
    else
        echo "#SBATCH --ntasks=1"
    fi
    [ -n "${SB_ACCOUNT:-}" ] && echo "#SBATCH --account=$SB_ACCOUNT"
    echo
    echo 'set -uo pipefail'
    echo 'cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1'
    echo 'echo "node: $(hostname)   job: ${SLURM_JOB_ID:-?}"'
    [ "$GPUS" -gt 0 ] && echo 'echo "gpus: ${CUDA_VISIBLE_DEVICES:-none}"  # read-only'
    echo
    echo "ENV_NAME=\"\${ENV_NAME:-${ENV_NAME:-rasp}}\""
    echo 'if command -v conda >/dev/null 2>&1; then'
    echo '    eval "$(conda shell.bash hook)" && conda activate "$ENV_NAME"'
    echo 'elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi'
    echo
    echo '# Compute nodes may have no outbound network; an online wandb run'
    echo '# can hang on connection retry. Sync later from the login node.'
    echo 'export WANDB_MODE="${WANDB_MODE:-offline}"'
    echo
    if [ "$GPUS" -gt 1 ]; then
        echo "srun ${CMD[*]}"      # one task per GPU, as Lightning expects
    else
        echo "${CMD[*]}"
    fi
} > "$SCRIPT"

echo "partition : $PART        qos: $QOS"
echo "resources : ${GPUS} gpu, ${CPUS} cpu, ${MEM} ram, ${TIME}"
echo "command   : ${CMD[*]}"
echo "script    : $SCRIPT"
if [ "$DRY" = 1 ]; then
    echo; sed 's/^/  /' "$SCRIPT"; exit 0
fi
sbatch "$SCRIPT"
