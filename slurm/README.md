# Running this repo on MBZUAI DANA (Slurm)

Written against the MBZUAI HPC policy page
(`hpc.mbzuai.ac.ae/wiki/policies.html`). Where that page and this file
disagree, the page wins — check it, because policy changes.

## The rules this repo has to respect

| policy | what it means here |
|---|---|
| **No compute on login nodes.** Background monitoring terminates compute processes outside Slurm and may close the SSH session. | `setup_env.sh` submits verification instead of running it. |
| Login node slice: **8 cores, 1000 tasks, 16 GB soft / 32 GB hard**. | Enough for the rule-model tests and the data config. Not for `verify_env.py` (112M model) or anything that loads a dataset. |
| **`sbatch` for batch, `srun` for interactive. Do not use `salloc` as the standard method.** | Use `srun --pty` when you need a shell. Earlier revisions of this file suggested `salloc`; that was wrong. |
| Interactive/debugging must use **`--qos=gpu-debug-qos`**, for environment testing, dependency validation, GPU verification and short experiments. | `verify.sbatch` sets exactly this — it is a GPU-verification job. Training must use a production QoS instead. |
| **No idle/reservation jobs** (`sleep`), close unused sessions. | Do not hold a debug shell open between runs. |
| **Never touch `CUDA_VISIBLE_DEVICES`** or reach for GPUs Slurm did not give you. Violations can end in account revocation. | Nothing in this repo sets it. `devices=-1` means "all GPUs Slurm assigned", which is the correct way to ask. `verify.sbatch` only *prints* the variable. |
| **No `--exclusive`** without written approval. | Not used anywhere here. |
| DANA GPU: up to **8 concurrent jobs**, up to **12 concurrent GPUs**. | A seed sweep must not fan out past this. 5 seeds x 1 GPU is fine. |
| Remove obsolete checkpoints and failed-experiment output. | `cleanup_workspace.sh` in the repo root; this workspace hit 124 G once. |

## ⚠️ Lustre small-files, and why it applies to the chord pipeline

The policy asks you to contact `hpc.admins@mbzuai.ac.ae` **before** large-scale
processing of very many small files on `/l/`, because of metadata load.

`run_pipeline_pop909_w1_orch.sh` is exactly that workload. It writes, all
under `/l/users/<you>/data/pop909_ivvi_w1/`:

- `snippets/` — one `.mid` per matched 4-bar window
- `snippets_perkey/` — **×12**, one per transposed key
- `orch_perkey/` — one folder per snippet, several `arrangement_band-*.mid` each

That is tens of thousands of small files before a single tensor is written.
Mail the HPC team first, and consider the mitigations the policy suggests —
aggregation, archive formats, or staging the intermediates on node-local
scratch (`$TMPDIR`) and only writing the final `.pt` files to `/l/`.

The `.pt` datasets and checkpoints are large and few, which is the pattern
Lustre wants. It is the MIDI intermediates that are the problem.

## Partition and QoS are PAIRED — mismatching them gets the job rejected

| workload | partition | QoS | limits |
|---|---|---|---|
| large / general GPU | `long` | `gpu-12` | **12 GPUs, 8 jobs, 72 h** |
| standard GPU | `cscc-gpu-p` | `cscc-gpu-qos` | 4 GPUs/account, 2 node-jobs, 24 h |
| interactive / debugging | `cscc-gpu-p` | `gpu-debug-qos` | short work only |
| CPU only | `cscc-cpu-p` | `cscc-cpu-qos` | 512 cores, 2 jobs, 72 h |

Minimum GPU request on either GPU partition is 1. `long` has 39 nodes,
`cscc-gpu-p` 16, `cscc-cpu-p` 16 (71 total). The `cscc` prefix is a legacy
SLURM identifier; all three are DANA.

## Submitting

Every job script is self-contained: the command lives inside it, and you
submit it with nothing after the filename, as in the DANA examples.

```bash
sbatch slurm/verify.sbatch                  # check the environment
sbatch slurm/train_chord_adapter.sbatch     # the flagship chord run
```

| script | partition + QoS | resources |
|---|---|---|
| `verify.sbatch` | `cscc-gpu-p` + `gpu-debug-qos` | 1 GPU, 8 cpu, 40 G, 20 min |
| `train_chord_adapter.sbatch` | `long` + `gpu-12` | 1 GPU, 8 cpu, 64 G, 24 h |
| `template_gpu.sbatch` | `long` + `gpu-12` | copy, edit, submit |
| `template_cpu.sbatch` | `cscc-cpu-p` + `cscc-cpu-qos` | copy, edit, submit |

For anything new, copy a template and put the command at the bottom:

```bash
cp slurm/template_gpu.sbatch slurm/my_job.sbatch
$EDITOR slurm/my_job.sbatch
sbatch slurm/my_job.sbatch
```

### One file to edit for a new cluster

Every script sources `slurm/env.sh`, which activates the conda env, sets
`WANDB_MODE=offline`, prints the node/GPU/paths, and exports:

| var | default |
|---|---|
| `RASP_REPO` | the repo containing `slurm/env.sh` |
| `D` | `$RASP_REPO/../data/pop909_ivvi_w1` |
| `BASE` | `checkpoints/cp_transformer_v0.42_..._size1_...ckpt` |

Those three are read as overridable defaults by every `midi_adapter/run_*.sh`
(they used to be hardcoded to `/l/users/xinyue.li/...`, which is why nothing
ran elsewhere). Fix `env.sh` once and every run script follows.

`--mail-type` / `--mail-user` are unavailable on DANA and are not used.
`--exclusive` needs approval and is not used.

## Four things that bite specifically under Slurm

### 1. wandb, on a node with no outbound network

The trainers import `WandbLogger` unconditionally, so **wandb must be
installed even when you disable it**. Disabling is separate from the import:

```bash
--wandb_project ''        # no WandbLogger; TensorBoard still logs
```

If you want wandb and compute nodes have no internet, run offline and sync
from the login node afterwards (syncing is file IO, so it is login-safe):

```bash
export WANDB_MODE=offline       # in the job script
wandb sync wandb/offline-run-*  # later
```

Without this a job can hang on a connection retry rather than fail outright.

### 2. Multi-GPU needs `srun`, not bare `python`

Both trainers use `devices=-1` and switch to `DDPStrategy` above one GPU.
Lightning reads Slurm from the environment, and the launch styles are not
interchangeable:

```bash
# WRONG: one process sees N GPUs, Lightning expects N tasks
#SBATCH --gres=gpu:4
python -m midi_adapter.train_cp_yinyang ...

# RIGHT: one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
srun python -m midi_adapter.train_cp_yinyang ...
```

Single GPU with one task is unaffected. Stay within the 12-GPU cap.

### 3. Wall time, preemption and requeue

**MaxWall is 72 h on `long`, 24 h on `cscc-gpu-p`.** Against the runs here:

| run | steps |
|---|---|
| chord adapter (`run_adapter_tracr.sh`) | 40,000 (8,000 for the short variants) |
| CP base pretrain (`pretrain_cp_bass.py`) | 200,000 |

72 h is generous for the adapter runs. The 200k-step CP base pretrain is
the one at risk, and if it does not fit, checkpoint-and-requeue is
mandatory rather than optional. Lightning's `SLURMEnvironment` auto-requeues on `SIGUSR1`, which
only helps if the job can resume:

```bash
#SBATCH --signal=SIGUSR1@90
python -m midi_adapter.train_cp_yinyang ... --resume_ckpt <path>
```

### 4. Checkpoints must land on shared storage

`--ckpt_dir` is relative to the submit directory. If that resolves to
node-local disk the checkpoints vanish with the job. Point it at `/l/`
explicitly — and then prune, per the data-management policy.

## What is safe on a login node

Within the 8-core / 16 GB slice, these are seconds and touch no dataset:

```bash
python -m models.test_integer_rule_model --max_seq_len 32
python -m data.integer_dataset
python -m models.integer_rule_model --build --max_seq_len 32   # Tracr, CPU
bash slurm/discover.sh
wandb sync wandb/offline-run-*
```

`verify_env.py --quick` skips the 112M forward pass and would fit the slice
too — but it then stops checking the thing most likely to be broken, so
prefer the job.

Everything else — any training, any evaluation, anything that loads a `.pt`
dataset, the orchestration pipeline — goes through `sbatch` or `srun`.
