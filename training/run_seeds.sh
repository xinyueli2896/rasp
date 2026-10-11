#!/usr/bin/env bash
# =============================================================================
#  Seed sweep for the integer experiment, one job per GPU in parallel.
#
#    bash training/run_seeds.sh                      # every row, seeds 0-4
#    bash training/run_seeds.sh adapter lora trainar
#    bash training/run_seeds.sh finetune
#    SEEDS="0 1 2" GPUS="0 1 2" bash training/run_seeds.sh
#    DRY=1 bash training/run_seeds.sh                # print the plan only
#
#  Rows:
#    pretrain   AR on pool A                        -> the baseline
#    finetune   AR continued on pool B, no adapter  -> forgetting control
#    adapter    cross-attention, AR FROZEN          -> the proposed model
#    lora       cross-attention + LoRA on the AR    -> partly trainable
#    trainar    cross-attention + AR fully trainable
#    enc        NO rule input: encoder -> rule embedding, unsupervised
#    encsup     the same with CE supervision on the encoder
#    scratch    AR from random init on A u B, no rule model   (one stage)
#    joint      AR from random init + adapter, jointly, A u B (one stage)
#
#  SCHEDULING
#  ----------
#  Two phases, because there is exactly one dependency: everything else for
#  seed N reads ar_integer_seed<N>.pt, so every pretrain must finish first.
#  Within a phase the jobs are independent and run concurrently, one per GPU.
#
#    phase 1   pretrain x |SEEDS|
#    ---- barrier ----
#    phase 2   (finetune, adapter, lora, trainar) x |SEEDS|
#
#  With 5 seeds, 3 adapter variants and 8 GPUs that is 15 phase-2 jobs in
#  two waves rather than 15 sequential runs.
#
#  Each job gets ONE GPU via --device cuda:N. These are plain PyTorch
#  scripts at d_model=128, so a second GPU would do nothing for a single
#  run; the parallelism is across runs. CUDA_VISIBLE_DEVICES is deliberately
#  left alone -- some clusters forbid touching it, and --device is both
#  sufficient and clearer.
#
#  Output goes to one log per job, since parallel stdout interleaves.
#  Then:  python -m training.results_table
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

SEEDS="${SEEDS:-0 1 2 3 4}"
LORA_RANK="${LORA_RANK:-16}"
LOGDIR="${LOGDIR:-run_logs}"
DRY="${DRY:-0}"

# GPUs: honour $GPUS, else every visible device, else CPU.
if [ -n "${GPUS:-}" ]; then
    read -r -a GPU <<<"$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    read -r -a GPU <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
else
    GPU=()
fi
[ ${#GPU[@]} -eq 0 ] && GPU=(cpu)

ROWS=("$@"); [ ${#ROWS[@]} -eq 0 ] && ROWS=(pretrain finetune adapter lora trainar scratch joint)
PROXY_W="${PROXY_W:-1.0}" 
want() { for r in "${ROWS[@]}"; do [ "$r" = "$1" ] && return 0; done; return 1; }

mkdir -p "$LOGDIR"
dev_of() { [ "$1" = cpu ] && echo cpu || echo "cuda:$1"; }

# ── run a list of "name<TAB>command" jobs, at most one per GPU ───────────
run_phase() {
    local phase="$1"; shift
    local -a jobs=("$@")
    [ ${#jobs[@]} -eq 0 ] && return 0
    echo
    echo "══ $phase: ${#jobs[@]} job(s) over ${#GPU[@]} device(s) ══"

    local i=0 rc=0
    while [ $i -lt ${#jobs[@]} ]; do
        local -a pids=() names=()
        for g in "${GPU[@]}"; do
            [ $i -lt ${#jobs[@]} ] || break
            local name="${jobs[$i]%%	*}" cmd="${jobs[$i]#*	}"
            local log="$LOGDIR/${name}.log"
            printf '  %-34s %-9s -> %s\n' "$name" "$(dev_of "$g")" "$log"
            if [ "$DRY" = 1 ]; then
                echo "      $cmd --device $(dev_of "$g")"
            else
                ( $cmd --device "$(dev_of "$g")" ) >"$log" 2>&1 &
                pids+=($!); names+=("$name")
            fi
            i=$((i + 1))
        done
        [ "$DRY" = 1 ] && continue
        for k in "${!pids[@]}"; do
            if ! wait "${pids[$k]}"; then
                echo "  FAILED: ${names[$k]}  (see $LOGDIR/${names[$k]}.log)" >&2
                tail -5 "$LOGDIR/${names[$k]}.log" | sed 's/^/      /' >&2
                rc=1
            fi
        done
    done
    return $rc
}

# ── phase 1: pretrain (everything else depends on it) ────────────────────
P1=()
if want pretrain; then
    for s in $SEEDS; do
        P1+=("pretrain_seed${s}	python -m training.pretrain_integer --seed $s")
    done
fi
run_phase "phase 1  AR pretrain" "${P1[@]}" || exit 1

# ── phase 2: everything that reads an AR checkpoint ──────────────────────
P2=()
for s in $SEEDS; do
    AR="checkpoints/ar_integer_seed${s}.pt"
    if want finetune || want adapter || want lora || want trainar; then
        if [ "$DRY" != 1 ] && [ ! -f "$AR" ]; then
            echo "missing $AR -- run the pretrain row for seed $s first:" >&2
            echo "  SEEDS=$s bash training/run_seeds.sh pretrain" >&2
            exit 1
        fi
    fi
    B="python -m training.finetune_adapter_integer --seed $s --ar_ckpt $AR"
    want finetune && P2+=("finetune_seed${s}	$B --mode finetune")
    want adapter  && P2+=("adapter_frozen_seed${s}	$B --mode adapter")
    want lora     && P2+=("adapter_lora${LORA_RANK}_seed${s}	$B --mode adapter --lora_rank $LORA_RANK")
    want trainar  && P2+=("adapter_trainar_seed${s}	$B --mode adapter --train_ar")
    want enc      && P2+=("adapter_enc_seed${s}	$B --mode adapter --rule_input encoder")
    want encsup   && P2+=("adapter_encsup_seed${s}	$B --mode adapter --rule_input encoder --proxy_loss_weight $PROXY_W")
    # scratch runs need no AR checkpoint: they build one.
    S="python -m training.finetune_adapter_integer --seed $s"
    want scratch  && P2+=("scratch_seed${s}	$S --mode scratch")
    want joint    && P2+=("joint_scratch_seed${s}	$S --mode joint_scratch")
done
run_phase "phase 2  finetune / adapter variants" "${P2[@]}" || exit 1

echo
echo "done. Logs in $LOGDIR/. Build the table with:"
echo "  python -m training.results_table"
