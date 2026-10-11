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
#  The only dependency is WITHIN a seed: everything else for seed N reads
#  ar_integer_seed<N>.pt. Two ways to exploit that, and which is faster
#  depends on the shape of the sweep:
#
#  PHASE mode (default). Two waves with a barrier:
#       phase 1   pretrain x |SEEDS|
#       ---- barrier ----
#       phase 2   (finetune, adapter, lora, ...) x |SEEDS|
#     Best when phase 2 has many more jobs than phase 1, because every GPU
#     is busy during the long phase.
#
#  PIPELINE mode (PIPELINE=1). One chain per seed, chains in parallel:
#       gpu0: pretrain_s0 -> finetune_s0 -> adapter_s0 -> ...
#       gpu1: pretrain_s1 -> finetune_s1 -> adapter_s1 -> ...
#     Best when |SEEDS| <= |GPUS| and each seed has few rows: seed N's
#     adapter starts the moment seed N's pretrain lands instead of waiting
#     for the slowest pretrain. With 5 seeds and 2 rows it removes the
#     barrier entirely; with 5 seeds and 5 rows the phase mode wins,
#     because the pipeline leaves 3 of 8 GPUs idle throughout.
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
PIPELINE="${PIPELINE:-0}"

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

# ── pipeline mode: one chain per seed, chains concurrent ─────────────────
if [ "$PIPELINE" = 1 ]; then
    echo
    echo "══ pipeline: one chain per seed over ${#GPU[@]} device(s) ══"
    pids=(); names=(); i=0
    for s in $SEEDS; do
        g="${GPU[$((i % ${#GPU[@]}))]}"; dev="$(dev_of "$g")"; i=$((i+1))
        AR="checkpoints/ar_integer_seed${s}.pt"
        B="python -m training.finetune_adapter_integer --seed $s --ar_ckpt $AR"
        chain=()
        want pretrain && chain+=("python -m training.pretrain_integer --seed $s")
        want finetune && chain+=("$B --mode finetune")
        want adapter  && chain+=("$B --mode adapter")
        want lora     && chain+=("$B --mode adapter --lora_rank $LORA_RANK")
        want trainar  && chain+=("$B --mode adapter --train_ar")
        want enc      && chain+=("$B --mode adapter --rule_input encoder")
        want encsup   && chain+=("$B --mode adapter --rule_input encoder --proxy_loss_weight $PROXY_W")
        want scratch  && chain+=("python -m training.finetune_adapter_integer --seed $s --mode scratch")
        want joint    && chain+=("python -m training.finetune_adapter_integer --seed $s --mode joint_scratch")
        [ ${#chain[@]} -eq 0 ] && continue
        printf '  seed %-3s %-9s %d step(s) -> %s/chain_seed%s.log\n' \
               "$s" "$dev" "${#chain[@]}" "$LOGDIR" "$s"
        if [ "$DRY" = 1 ]; then
            for c in "${chain[@]}"; do echo "      $c --device $dev"; done
            continue
        fi
        (
            for c in "${chain[@]}"; do
                echo "### $c" >>"$LOGDIR/chain_seed${s}.log"
                $c --device "$dev" >>"$LOGDIR/chain_seed${s}.log" 2>&1 || exit 1
            done
        ) &
        pids+=($!); names+=("seed$s")
    done
    rc=0
    for k in "${!pids[@]}"; do
        if ! wait "${pids[$k]}"; then
            echo "  FAILED chain ${names[$k]} (see $LOGDIR/chain_${names[$k]}.log)" >&2
            rc=1
        fi
    done
    echo; echo "done. Build the table with:  python -m training.results_table"
    exit $rc
fi

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
