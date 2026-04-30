#!/usr/bin/env python3
"""
Full experiment pipeline.

Pretrain : AR on {0..6}
Finetune : 3 variants (2..16 | 0..16 | 0..16+{18,22,23})
Adapters : 3 variants × 5 seeds, no LoRA, frozen AR, n_skip=1, 400 epochs
Evaluate : all checkpoints
"""

import os, sys, subprocess

BASE    = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'data', 'dataset.py')
AR_CKPT = 'checkpoints/ar_pretrain_0to6.pt'


# ---------------------------------------------------------------------------
# dataset.py patcher
# ---------------------------------------------------------------------------

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
    patch(
        ar='list(range(6))',
        ft='list(range(2, 16))',
        adapter='FINETUNE_STARTERS',
    )
    print('[dataset.py reverted]')


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def ckpt_exists(name):
    return os.path.exists(os.path.join(BASE, 'checkpoints', f'{name}.pt'))


def run(cmd, desc, ckpt_name=None):
    if ckpt_name and ckpt_exists(ckpt_name):
        print(f'\n[SKIP] {desc}  ({ckpt_name}.pt already exists)')
        return
    print(f'\n{"="*70}', flush=True)
    print(f'{desc}', flush=True)
    print(f'{"="*70}', flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=BASE)
    if r.returncode != 0:
        print(f'\n[FAILED exit={r.returncode}] {desc}')
        revert()
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Pretrain AR on {0..6}
# ---------------------------------------------------------------------------

patch(ar='list(range(7))')
run(
    ['training/pretrain.py', '--epochs', '200', '--ckpt_name', 'ar_pretrain_0to6'],
    'PRETRAIN  starters={0..6}  →  ar_pretrain_0to6.pt',
    ckpt_name='ar_pretrain_0to6',
)

# ---------------------------------------------------------------------------
# 2. Finetune (3 variants)
# ---------------------------------------------------------------------------

FINETUNE_CONFIGS = [
    ('list(range(2, 17))',                 'ar_finetuned_2to16',      '{2..16}'),
    ('list(range(0, 17))',                 'ar_finetuned_0to16',      '{0..16}'),
    ('list(range(0, 17)) + [18, 22, 23]', 'ar_finetuned_0to16_plus', '{0..16,18,22,23}'),
]

for ft_expr, ckpt_name, label in FINETUNE_CONFIGS:
    patch(ft=ft_expr)
    run(
        ['training/finetune.py',
         '--epochs', '200',
         '--ar_ckpt', AR_CKPT,
         '--ckpt_name', ckpt_name],
        f'FINETUNE  starters={label}  →  {ckpt_name}.pt',
        ckpt_name=ckpt_name,
    )

# ---------------------------------------------------------------------------
# 3. Adapter training (3 variants × 5 seeds, no LoRA)
# ---------------------------------------------------------------------------

ADAPTER_CONFIGS = [
    ('list(range(2, 17))',                 'adapter_2to16',      '{2..16}'),
    ('list(range(0, 17))',                 'adapter_0to16',      '{0..16}'),
    ('list(range(0, 17)) + [18, 22, 23]', 'adapter_0to16_plus', '{0..16,18,22,23}'),
]

for adapter_expr, stem, label in ADAPTER_CONFIGS:
    patch(adapter=adapter_expr)
    # Skip only if all 5 seeds already exist
    all_seeds_done = all(ckpt_exists(f'{stem}_seed{i}') for i in range(5))
    run(
        ['training/train_yinyang_multirun.py',
         '--n_seeds', '5', '--ckpt_stem', stem,
         '--', '--n_skip', '1', '--no_lora', '--epochs', '400',
         '--ar_ckpt', AR_CKPT],
        f'ADAPTER   starters={label}  →  {stem}_seed{{0..4}}.pt',
        ckpt_name=f'{stem}_seed4' if all_seeds_done else None,
    )

# ---------------------------------------------------------------------------
# 4. Revert dataset.py
# ---------------------------------------------------------------------------

revert()

# ---------------------------------------------------------------------------
# 5. Evaluate all configurations
# ---------------------------------------------------------------------------

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
