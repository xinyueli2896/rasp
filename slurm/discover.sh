#!/usr/bin/env bash
# =============================================================================
#  Ask the scheduler what this cluster allows, so the job scripts can be set
#  from fact rather than from a wiki page or a guess.
#
#  Pure scheduler queries -- sinfo / sacctmgr / scontrol. No compute, nothing
#  submitted, safe on a login node.
#
#    bash slurm/discover.sh                 # human-readable
#    bash slurm/discover.sh > cluster.txt   # paste this back
# =============================================================================

set -uo pipefail

have() { command -v "$1" >/dev/null 2>&1; }
hdr()  { echo; echo "═══ $* ═══"; }

if ! have sinfo; then
    echo "No sinfo on PATH — is this a Slurm login node?" >&2
    exit 1
fi

echo "cluster : $(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/{print $2}' | xargs || echo '?')"
echo "user    : $USER"
echo "host    : $(hostname)"

hdr "partitions (NAME AVAIL TIMELIMIT NODES STATE)"
sinfo -o '%20P %.9a %.14l %.6D %.6t %N' 2>/dev/null

hdr "per-partition limits (default/max time, max nodes, max cpus)"
printf '%-20s %-14s %-14s %-10s %-10s\n' PARTITION DEF_TIME MAX_TIME MAX_NODES MAX_CPUS
for p in $(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u); do
    info=$(scontrol show partition "$p" 2>/dev/null)
    get() { echo "$info" | grep -oE "$1=[^ ]+" | head -1 | cut -d= -f2; }
    printf '%-20s %-14s %-14s %-10s %-10s\n' "$p" \
        "$(get DefaultTime)" "$(get MaxTime)" "$(get MaxNodes)" "$(get MaxCPUsPerNode)"
done

hdr "GPU types and counts per node"
sinfo -h -o '%20P %20N %10c %10m %40G' 2>/dev/null | sort -u

hdr "your associations: accounts, QOS, limits"
if have sacctmgr; then
    sacctmgr -p show assoc user="$USER" \
        format=Cluster,Account,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit,MaxWall 2>/dev/null \
        | column -t -s'|'
else
    echo "sacctmgr unavailable"
fi

hdr "QOS definitions (limits that actually bind)"
if have sacctmgr; then
    sacctmgr -p show qos \
        format=Name,Priority,MaxWall,MaxTRES,MaxTRESPU,MaxJobsPU,MaxSubmitJobsPU,GrpTRES 2>/dev/null \
        | column -t -s'|'
else
    echo "sacctmgr unavailable"
fi

hdr "default account"
if have sacctmgr; then
    sacctmgr -n -p show user "$USER" format=DefaultAccount 2>/dev/null | tr -d '|'
fi

hdr "your current jobs"
squeue -u "$USER" -o '%.10i %.12P %.20j %.8T %.10M %.6D %R' 2>/dev/null || true

hdr "what the job scripts need from the above"
cat <<'NOTE'
  SB_PARTITION  a partition from the first table that has GPUs
  SB_ACCOUNT    your default account, unless you must bill a project
  SB_GRES       the gres syntax this cluster uses, from the GPU table
                (gpu:1, or gpu:a100:1, or similar -- it differs per site)
  SB_QOS        only if a QOS above is required rather than optional

  Also note MaxWall: slurm/verify.sbatch asks for 20 minutes, and the
  training jobs will want far more. If MaxWall is shorter than a training
  run, checkpoint-and-requeue is not optional -- see slurm/README.md.
NOTE
