#!/usr/bin/env python3
"""Finish the interrupted run_remaining.py: adapter_0to16 seeds 2-4, then adapter_0to16_plus x5."""
import os, sys, subprocess

BASE    = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'data', 'dataset.py')
AR_CKPT = 'checkpoints/ar_pretrain_0to6.pt'

def patch(adapter=None):
    with open(DATASET) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        s = line.strip()
        if adapter is not None and s.startswith('ADAPTER_TRAIN_STARTERS') and '=' in s:
            out.append(f'ADAPTER_TRAIN_STARTERS = {adapter}\n')
        else:
            out.append(line)
    with open(DATASET, 'w') as f:
        f.writelines(out)

def revert():
    patch(adapter='FINETUNE_STARTERS')
    print('[dataset.py reverted]')

def run(cmd, desc):
    print(f'\n{"="*70}\n{desc}\n{"="*70}', flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=BASE)
    if r.returncode != 0:
        print(f'\n[FAILED exit={r.returncode}] {desc}')
        revert()
        sys.exit(1)

# adapter_0to16 seeds 2-4 (0 and 1 already exist)
patch(adapter='list(range(0, 17))')
for seed in [2, 3, 4]:
    run(
        ['training/train_yinyang.py',
         '--seed', str(seed),
         '--ckpt_name', f'adapter_0to16_seed{seed}',
         '--n_skip', '1', '--no_lora', '--epochs', '400', '--ar_ckpt', AR_CKPT],
        f'ADAPTER  adapter_0to16  seed={seed}',
    )

# adapter_0to16_plus x5 seeds
patch(adapter='list(range(0, 17)) + [18, 22, 23]')
for seed in range(5):
    run(
        ['training/train_yinyang.py',
         '--seed', str(seed),
         '--ckpt_name', f'adapter_0to16_plus_seed{seed}',
         '--n_skip', '1', '--no_lora', '--epochs', '400', '--ar_ckpt', AR_CKPT],
        f'ADAPTER  adapter_0to16_plus  seed={seed}',
    )

revert()
print('\n=== REMAINING ADAPTER RUNS COMPLETE ===')
