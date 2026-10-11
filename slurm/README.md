# Running this repo on a Slurm cluster

## First: find out what your cluster allows

The `#SBATCH` defaults in `verify.sbatch` are a placeholder. Partition names,
account strings, QOS requirements and gres syntax differ at every site, so
ask the scheduler rather than guessing:

```bash
bash slurm/discover.sh          # pure sinfo/sacctmgr queries, login-safe
```

It prints the partitions, per-partition time limits, GPU types and counts,
your accounts and QOS limits, and says which of `SB_PARTITION`,
`SB_ACCOUNT`, `SB_GRES`, `SB_QOS` to set from them. Watch `MaxWall` in
particular: if it is shorter than a training run, checkpoint-and-requeue
stops being optional (see Preemption below).

Nothing heavier than `pip install` belongs on a login node. `setup_env.sh`
enforces that: on a machine where `sbatch` exists and `$SLURM_JOB_ID` is
unset, it installs and then **submits** the verification instead of running
it, because `verify_env.py` builds the 112M CP transformer.

```bash
bash setup_env.sh                 # install on the login node, submit verify
squeue -u "$USER"
cat slurm_logs/verify-<jobid>.out # expect: environment OK
```

Override the resources to match your cluster:

```bash
SB_PARTITION=gpu SB_ACCOUNT=myproj SB_GRES=gpu:a100:1 bash setup_env.sh
```

`--local` runs verification in place — only do that inside an interactive
allocation (`salloc` / `srun --pty`), never on the login node.

---

## Four things that bite specifically under Slurm

### 1. wandb, on a node with no outbound network

The trainers import `WandbLogger` unconditionally, so **wandb must be
installed even when you disable it**. Disabling the logger is separate from
the import:

```bash
--wandb_project ''        # no WandbLogger at all; TensorBoard still logs
```

If you do want wandb and the compute nodes have no internet — common — run
offline and sync afterwards from the login node:

```bash
export WANDB_MODE=offline     # in the job script
wandb sync wandb/offline-run-*  # later, from a node with network
```

Without this a job can hang on a connection retry rather than fail outright.

### 2. Multi-GPU needs `srun`, not bare `python`

Both trainers use `devices=-1` and switch to `DDPStrategy` when more than one
GPU is visible. Lightning detects Slurm from the environment, and the two
launch styles are not interchangeable:

```bash
# WRONG for multi-GPU: one process sees N GPUs, Lightning expects N tasks
#SBATCH --gres=gpu:4
python -m midi_adapter.train_cp_yinyang ...

# RIGHT: one task per GPU, srun launches them
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
srun python -m midi_adapter.train_cp_yinyang ...
```

Single GPU (`--gres=gpu:1`, one task) is unaffected — plain `python` is fine.

### 3. Preemption and requeue

Lightning's `SLURMEnvironment` auto-requeues on `SIGUSR1` by default. On a
preemptible partition that is usually what you want, but it only helps if the
run can resume, so pass `--resume_ckpt` and give the scheduler a signal
window:

```bash
#SBATCH --signal=SIGUSR1@90
```

If you would rather a preempted job just die, set
`SLURM_EXPORT_ENV` aside and disable it in code — but note that
`train_cp_yinyang.py` does not currently construct `SLURMEnvironment`
explicitly, so it takes Lightning's default.

### 4. Checkpoints and logs go to shared storage

`--ckpt_dir` defaults to `checkpoints/` relative to the submit directory. If
that lands on a node-local disk, the files vanish when the job ends. Point it
at project/scratch space explicitly, and remember the checkpoints are large —
`cleanup_workspace.sh` exists because this got to 124 G once.

---

## What does NOT need a GPU

These are small enough for a login node, or for `--cpus-per-task=2`:

| | why |
|---|---|
| `python -m models.test_integer_rule_model` | 75-dim model, 32 checks, seconds |
| `python -m data.integer_dataset` | prints the split, no tensors of size |
| `python -m models.integer_rule_model --build` | Tracr compile, CPU-only, needs `requirements-compile.txt` |

`verify_env.py --quick` skips the 112M forward pass and is also login-safe,
but it then stops checking the thing most likely to be broken.
