#!/usr/bin/env bash
# =============================================================================
#  Check every slurm/*.sbatch against DANA policy, before the scheduler does.
#
#    bash slurm/lint.sh
#
#  Encodes the documented rules so a new job script cannot quietly break
#  them. Pure text inspection -- nothing is submitted, safe on a login node.
#
#  Rules checked (hpc.mbzuai.ac.ae/wiki):
#    1  partition and QoS must be a documented PAIR
#    2  wall time within the partition's cap
#    3  GPU count within the partition's cap, and >= 1 on a GPU partition
#    4  CPU cores <= 512 on cscc-cpu-p
#    5  ONLY the directives the DANA example uses. --error in particular is
#       not among them; Slurm merges stderr into --output without it.
#    6  no --exclusive, no --mail-type / --mail-user
#    7  --output directory must exist in a fresh clone: Slurm opens it
#       BEFORE the script runs, so an in-script mkdir is too late
#    8  no process may set CUDA_VISIBLE_DEVICES
#    9  nothing may `sleep` to hold an allocation
# =============================================================================

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

FAIL=0
bad() { echo "  FAIL  $1"; FAIL=1; }
ok()  { echo "  ok    $1"; }

# partition -> "qos1,qos2 maxhours maxgpu"
caps() {
    case "$1" in
        long)       echo "gpu-12 72 12" ;;
        cscc-gpu-p) echo "cscc-gpu-qos,gpu-debug-qos 24 4" ;;
        cscc-cpu-p) echo "cscc-cpu-qos 72 0" ;;
        *)          echo "" ;;
    esac
}

for f in slurm/*.sbatch; do
    echo "$(basename "$f")"
    d() { grep -oP "(?<=^#SBATCH --$1=)\S+" "$f" | head -1; }

    PART=$(d partition); QOS=$(d qos); TIME=$(d time)
    GRES=$(d gres);      CPUS=$(d cpus-per-task); OUT=$(d output)
    NT=$(d ntasks-per-node)

    # 1. pairing
    info=$(caps "$PART")
    if [ -z "$info" ]; then
        bad "unknown partition '${PART:-<unset>}'"
        continue
    fi
    read -r QOSLIST MAXH MAXGPU <<<"$info"
    if [[ ",$QOSLIST," == *",$QOS,"* ]]; then
        ok "$PART + $QOS is a documented pair"
    else
        bad "$PART does not take qos '$QOS' (valid: $QOSLIST)"
    fi

    # 2. wall time
    hrs=$(( 10#$(echo "${TIME:-0}" | cut -d: -f1) ))
    if [ "$hrs" -le "$MAXH" ]; then ok "time $TIME within ${MAXH}h"
    else bad "time $TIME exceeds the ${MAXH}h cap on $PART"; fi

    # 3. GPUs
    N=$(echo "${GRES:-}" | grep -oP '\d+$' || true)
    if [ "$MAXGPU" = 0 ]; then
        [ -z "$GRES" ] && ok "no GPU on a CPU partition" \
                       || bad "--gres on the CPU partition $PART"
    elif [ -z "${N:-}" ]; then
        bad "$PART has a minimum request of 1 GPU, none requested"
    elif [ "$N" -le "$MAXGPU" ]; then
        ok "gpu:$N within the $MAXGPU-GPU cap"
    else
        bad "gpu:$N exceeds the $MAXGPU-GPU cap on $PART"
    fi

    # multi-GPU needs one task per GPU, and srun to launch them
    if [ -n "${N:-}" ] && [ "$N" -gt 1 ]; then
        if [ "${NT:-0}" = "$N" ] && grep -qE '^\s*srun ' "$f"; then
            ok "multi-GPU uses --ntasks-per-node=$N and srun"
        else
            bad "gpu:$N needs --ntasks-per-node=$N AND an srun launch " \
                "(Lightning uses devices=-1 and expects one task per GPU)"
        fi
    fi

    # 4. CPU cap
    if [ "$PART" = cscc-cpu-p ] && [ "${CPUS:-0}" -gt 512 ]; then
        bad "cpus-per-task=$CPUS exceeds the 512-core cap"
    fi

    # 5. only the directives the DANA example uses
    LEGAL=" job-name output nodes ntasks cpus-per-task mem gres partition qos time "
    offenders=""
    while read -r k; do
        [[ "$LEGAL" == *" $k "* ]] || offenders="$offenders --$k"
    done < <(grep -oP '(?<=^#SBATCH )--\K[a-z-]+' "$f")
    if [ -z "$offenders" ]; then
        ok "only the documented directives"
    else
        bad "non-standard directive(s):$offenders"
    fi

    # 6. forbidden / unavailable flags (also caught above, named explicitly)
    for flag in exclusive mail-type mail-user; do
        grep -q "^#SBATCH --$flag" "$f" && bad "--$flag must not be used"
    done

    # 7. the log directory has to exist before the job starts
    dir=$(dirname "${OUT:-slurm_logs/x}")
    if [ -d "$dir" ] && git ls-files --error-unmatch "$dir" >/dev/null 2>&1; then
        ok "--output dir '$dir' is tracked, so it exists in a fresh clone"
    elif [ -d "$dir" ] && ls -A "$dir"/.gitkeep >/dev/null 2>&1 \
         && git ls-files --error-unmatch "$dir/.gitkeep" >/dev/null 2>&1; then
        ok "--output dir '$dir' kept by a tracked .gitkeep"
    else
        bad "--output dir '$dir' is absent from a fresh clone; Slurm opens " \
            "it before the script runs, so the job dies with no log"
    fi

    # 8-9. runtime rules
    grep -qE '^\s*(export\s+)?CUDA_VISIBLE_DEVICES=' "$f" \
        && bad "sets CUDA_VISIBLE_DEVICES (prohibited; read-only is fine)"
    grep -qE '^\s*sleep\s' "$f" \
        && bad "sleeps (idle reservation jobs are not permitted)"
    echo
done

if [ "$FAIL" = 0 ]; then echo "all job scripts conform."; else echo "policy violations above."; fi
exit $FAIL
