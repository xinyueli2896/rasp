#!/usr/bin/env python3
"""
Continue experiment pipeline from after finetune {2..16} (already done).
Skips: pretrain, finetune {2..16}
Runs:  finetune {0..16} (redo), finetune {0..16+}, adapters x3, evaluate
"""

import os, sys, subprocess

BASE    = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'data', 'dataset.py')
AR_CKPT = 'checkpoints/ar_pretrain_0to6.pt'


def patch(ar=None, ft=None, adapter=None):
    with open(DATASET) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        s = line.strip()
        if ar is not None and s.startswith('AR_TRAIN_STARTERS') and '=' in s:
            out.append(f'AR_TRAIN_STARTERS      = {ar}\n')
        elif ft is not None and s.startswith('FINETUNE_STARTERS') and '=' in s:
            out.append(f'FINETUNE_STARTERS      = {ft}\n')
        elif adapter is not None and s.startswith('ADAPTER_TRAIN_STARTERS') and '=' in s:
            out.append(f'ADAPTER_TRAIN_STARTERS = {adapter}\n')
        else:
            out.append(line)
    with open(DATASET, 'w') as f:
        f.writelines(out)


def revert():
    patch(ar='list(range(6))', ft='list(range(2, 16))', adapter='FINETUNE_STARTERS')
    print('[dataset.py reverted]')


def run(cmd, desc):
    print(f'\n{"="*70}', flush=True)
    print(f'{desc}', flush=True)
    print(f'{"="*70}', flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=BASE)
    if r.returncode != 0:
        print(f'\n[FAILED exit={r.returncode}] {desc}')
        revert()
        sys.exit(1)


# Finetune {0..16} (redo — died mid-run, checkpoint is best-so-far but redo for completeness)
patch(ft='list(range(0, 17))')
run(
    ['training/finetune.py', '--epochs', '200', '--ar_ckpt', AR_CKPT,
     '--ckpt_name', 'ar_finetuned_0to16'],
    'FINETUNE  starters={0..16}  ->  ar_finetuned_0to16.pt',
)

# Finetune {0..16, 18, 22, 23}
patch(ft='list(range(0, 17)) + [18, 22, 23]')
run(
    ['training/finetune.py', '--epochs', '200', '--ar_ckpt', AR_CKPT,
     '--ckpt_name', 'ar_finetuned_0to16_plus'],
    'FINETUNE  starters={0..16,18,22,23}  ->  ar_finetuned_0to16_plus.pt',
)

# Adapters x3 configs x5 seeds
ADAPTER_CONFIGS = [
    ('list(range(2, 17))',                 'adapter_2to16',      '{2..16}'),
    ('list(range(0, 17))',                 'adapter_0to16',      '{0..16}'),
    ('list(range(0, 17)) + [18, 22, 23]', 'adapter_0to16_plus', '{0..16,18,22,23}'),
]

for adapter_expr, stem, label in ADAPTER_CONFIGS:
    patch(adapter=adapter_expr)
    run(
        ['training/train_yinyang_multirun.py',
         '--n_seeds', '5', '--ckpt_stem', stem,
         '--', '--n_skip', '1', '--no_lora', '--epochs', '400',
         '--ar_ckpt', AR_CKPT],
        f'ADAPTER   starters={label}  ->  {stem}_seed{{0..4}}.pt',
    )

revert()

# Evaluate all
print(f'\n{"="*70}')
print('EVALUATION')
print(f'{"="*70}')

for ft_ckpt, ft_label in [
    ('ar_finetuned_2to16',      'ft=2to16'),
    ('ar_finetuned_0to16',      'ft=0to16'),
    ('ar_finetuned_0to16_plus', 'ft=0to16+'),
]:
    run(
        ['evaluate.py',
         '--ar_ckpt', AR_CKPT,
         '--ft_ckpt', f'checkpoints/{ft_ckpt}.pt',
         '--adapter_stems', 'adapter_2to16', 'adapter_0to16', 'adapter_0to16_plus',
         '--verbose'],
        f'EVAL  pretrain=ar_pretrain_0to6  {ft_label}  adapters=all3',
    )

print('\n\n=== ALL EXPERIMENTS COMPLETE ===')
