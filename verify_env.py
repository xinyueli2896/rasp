#!/usr/bin/env python3
"""Prove the environment can actually run this repo.

Importing a package is a weak test. The failures that cost real time here are
ones that import fine and break later:

  * `lightning` installed instead of `pytorch_lightning` — every module does
    `import pytorch_lightning as L`, and the newer package does not provide
    that name
  * a transformers bump moving `RoFormerEncoder`, which cp_transformer.py
    imports from a non-public path
  * torch installed without CUDA — trains at a fraction of the speed with no
    error anywhere
  * `wandb` missing — the trainers import WandbLogger unconditionally, so the
    run dies at import even with logging disabled

So this builds the real 112M CP transformer and pushes a batch through it,
and runs the rule model end to end.

    python verify_env.py
    python verify_env.py --quick      # skip the 112M forward pass
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FAILS: list[str] = []
WARNS: list[str] = []


def ok(name, cond, detail='', warn_only=False):
    tag = 'PASS' if cond else ('WARN' if warn_only else 'FAIL')
    print(f'  {tag}  {name}' + (f'   {detail}' if detail else ''))
    if not cond:
        (WARNS if warn_only else FAILS).append(name)
    return cond


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true',
                   help='skip the 112M-parameter forward pass')
    a = p.parse_args()

    print(f'python {sys.version.split()[0]}   repo {ROOT}')

    # -- 0. is the interpreter the one you think? -------------------------
    # `pip install` saying "Defaulting to user installation because normal
    # site-packages is not writeable" while a conda env is active means the
    # packages are going to ~/.local, not the env. They may still import --
    # but they shadow the env for EVERY python of that minor version, so a
    # later `conda activate other_env` silently inherits them.
    print('\n0. interpreter and where packages come from')
    import site
    print(f'  exe    {sys.executable}')
    print(f'  prefix {sys.prefix}')
    in_env = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    conda  = os.environ.get('CONDA_DEFAULT_ENV')
    usersite = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    print(f'  env    {"conda:" + conda if conda else ("venv" if in_env else "system python")}')
    if usersite and os.path.isdir(usersite):
        print(f'  user   {usersite}  (shadows the env for all python'
              f'{sys.version_info.major}.{sys.version_info.minor})')
    print('\n1. core packages')
    for mod, why in [('torch', 'everything'),
                     ('numpy', 'everything'),
                     ('pytorch_lightning', 'all trainers (NOT `lightning`)'),
                     ('transformers', 'the CP transformer'),
                     ('pretty_midi', 'MIDI IO'),
                     ('wandb', 'imported unconditionally by the trainers')]:
        try:
            m = importlib.import_module(mod)
            loc = getattr(m, '__file__', '') or ''
            tag = ' [user-site]' if '/.local/' in loc else ''
            ok(f'{mod:20s} {getattr(m, "__version__", "?")}{tag}', True, why)
        except ImportError as e:
            ok(f'{mod:20s} MISSING', False, f'{why} — {e}')

    # The specific mistake worth calling out by name.
    try:
        importlib.import_module('lightning')
        if 'pytorch_lightning' not in sys.modules:
            ok('`lightning` present but `pytorch_lightning` is not', False,
               'install pytorch_lightning; the code imports that name')
    except ImportError:
        pass

    # -- 2. the non-public import most likely to break ---------------------
    print('\n2. the fragile transformers import')
    try:
        from transformers.models.roformer.modeling_roformer import (  # noqa: F401
            RoFormerModel, RoFormerConfig, RoFormerEncoder)
        ok('RoFormerModel / RoFormerConfig / RoFormerEncoder', True,
           'cp_transformer.py:1 depends on this path')
    except Exception as e:
        ok('RoFormerModel / RoFormerConfig / RoFormerEncoder', False,
           f'{type(e).__name__}: {e}')

    # -- 3. hardware -------------------------------------------------------
    print('\n3. hardware')
    try:
        import torch
        cuda = torch.cuda.is_available()
        ok('CUDA available', cuda,
           f'{torch.cuda.device_count()} device(s): '
           f'{torch.cuda.get_device_name(0)}' if cuda
           else 'torch is CPU-only — it will run, just far slower. If this '
                'cluster has GPUs, reinstall torch from plain PyPI.',
           warn_only=True)
        if cuda:
            ok('a tensor actually reaches the GPU',
               float((torch.ones(8, device='cuda') * 2).sum()) == 16.0)
    except Exception as e:
        ok('torch hardware check', False, str(e))

    # -- 4. the rule model, end to end -------------------------------------
    print('\n4. rule model (committed weights, no tracr needed)')
    try:
        import torch
        from models.integer_rule_model import IntegerRuleModel, cache_path
        from rasp_program.next_token_rule import sequence, next_token
        pth = cache_path(32)
        if not os.path.exists(pth):
            ok(f'compiled weights at {pth}', False,
               'missing — `git pull`, or rebuild with '
               '`pip install -r requirements-compile.txt` then '
               '`python -m models.integer_rule_model --build --max_seq_len 32`')
        else:
            m = IntegerRuleModel.load(pth)
            seq = torch.as_tensor(sequence([9], 8))
            pred = m.from_tokens(seq)[1].argmax(-1)[0].numpy()
            ok('loads and predicts correctly',
               (pred == next_token([9], 8)[0]).all(),
               f'x=9 -> {pred.tolist()}')
            probs = torch.full((1, 6, 12), 1 / 12, requires_grad=True)
            m(probs)[1].sum().backward()
            ok('gradient flows through the frozen model',
               probs.grad is not None and float(probs.grad.abs().sum()) > 0)
    except Exception as e:
        ok('rule model', False, f'{type(e).__name__}: {e}')

    # -- 5. data config ----------------------------------------------------
    print('\n5. data config')
    try:
        from data.integer_dataset import get_dataloaders, validate_split
        info = validate_split()
        tr, ev = get_dataloaders('adapter', batch_size=8, n_repeats=4)
        x, y = next(iter(tr))
        ok('split validates and loaders build', True,
           f'seen={len(info["seen"])} unseen={len(info["unseen"])} '
           f'batch={tuple(x.shape)}')
    except Exception as e:
        ok('data config', False, f'{type(e).__name__}: {e}')

    # -- 6. the 112M CP transformer ----------------------------------------
    print('\n6. CP transformer (the chord/bass base)')
    if a.quick:
        print('  SKIP  (--quick)')
    else:
        try:
            import torch
            from cp_transformer import RoFormerSymbolicTransformer
            net = RoFormerSymbolicTransformer(size=1, max_lr=1e-4,
                                              with_velocity=False)
            n = sum(p.numel() for p in net.parameters())
            ok('builds', n > 1e8, f'{n:,} parameters')
            # A valid CP tuple is (program, pitch, duration, velocity).
            # preprocess() packs these as pitch + (dur+1)*128, and the
            # tokenizer allows 24 durations x 128 pitches, so duration must
            # stay under 24 or the embedding lookup goes out of range.
            x = torch.zeros(1, 4, 64, dtype=torch.long)
            x[..., 0::4] = torch.randint(0, 128, (1, 4, 16))   # program
            x[..., 1::4] = torch.randint(0, 128, (1, 4, 16))   # pitch
            x[..., 2::4] = torch.randint(0, 24,  (1, 4, 16))   # duration
            with torch.no_grad():
                xp = net.preprocess(x, torch.zeros(1, dtype=torch.long))
                out = net(xp)
            # local_decode flattens batch and sequence, so the leading dim
            # is B*seq_len, not B. What matters is that it ran, produced
            # per-token logits over the full vocab, and they are finite.
            ok('forward pass runs',
               out.shape[-1] == net.tokenizer.n_tokens
               and bool(torch.isfinite(out).all()),
               f'logits {tuple(out.shape)}, vocab {net.tokenizer.n_tokens}')
        except Exception as e:
            ok('CP transformer', False, f'{type(e).__name__}: {e}')

    # -- 7. optional extras -------------------------------------------------
    print('\n7. optional (absent is fine unless you need the feature)')
    for mod, why in [('tracr', 'recompiling rule models'),
                     ('jax', 'recompiling rule models'),
                     ('scipy', 'AccoMontage orchestration'),
                     ('peft', 'LoRA (--lora_rank > 0)')]:
        try:
            importlib.import_module(mod)
            print(f'  have  {mod:10s} ({why})')
        except ImportError:
            print(f'  --    {mod:10s} ({why})')

    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED: {FAILS}')
    else:
        print('environment OK' + (f'  ({len(WARNS)} warning(s): {WARNS})'
                                  if WARNS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
