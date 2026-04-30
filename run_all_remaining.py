#!/usr/bin/env python3
"""
Resilient continuation script — skips checkpoints that already exist.
Finishes: adapter_0to16 seeds 3-4, adapter_0to16_plus x5
Then runs: run_train_ar_experiments.sh
"""
import os, sys, subprocess

BASE    = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'data', 'dataset.py')
AR_CKPT = 'checkpoints/ar_pretrain_0to6.pt'

def patch(adapter):
    with open(DATASET) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith('ADAPTER_TRAIN_STARTERS') and '=' in s:
            out.append(f'ADAPTER_TRAIN_STARTERS = {adapter}\n')
        else:
            out.append(line)
    with open(DATASET, 'w') as f:
        f.writelines(out)

def revert():
    patch('FINETUNE_STARTERS')
    print('[dataset.py reverted]')

def run(cmd, desc):
    print(f'\n{"="*70}\n{desc}\n{"="*70}', flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=BASE)
    if r.returncode != 0:
        print(f'[FAILED exit={r.returncode}] {desc}')
        revert()
        sys.exit(1)

def ckpt_exists(name):
    return os.path.exists(os.path.join(BASE, 'checkpoints', f'{name}.pt'))

# adapter_0to16 seeds 3-4
patch('list(range(0, 17))')
for seed in [3, 4]:
    name = f'adapter_0to16_seed{seed}'
    if ckpt_exists(name):
        print(f'Skipping {name} (already exists)')
        continue
    run(['training/train_yinyang.py',
         '--seed', str(seed), '--ckpt_name', name,
         '--n_skip', '1', '--no_lora', '--epochs', '400', '--ar_ckpt', AR_CKPT],
        f'ADAPTER adapter_0to16 seed={seed}')

# adapter_0to16_plus x5
patch('list(range(0, 17)) + [18, 22, 23]')
for seed in range(5):
    name = f'adapter_0to16_plus_seed{seed}'
    if ckpt_exists(name):
        print(f'Skipping {name} (already exists)')
        continue
    run(['training/train_yinyang.py',
         '--seed', str(seed), '--ckpt_name', name,
         '--n_skip', '1', '--no_lora', '--epochs', '400', '--ar_ckpt', AR_CKPT],
        f'ADAPTER adapter_0to16_plus seed={seed}')

revert()
print('\n=== ALL DONE ===')
