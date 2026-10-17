"""
One accuracy table across every CP bass run, in the shape of the integer
experiment's evaluate.py output.

evaluate_cp_yinyang.py evaluates a single --adapter_ckpt and prints its own
table, so comparing six models means six tables to read side by side. This
loops the runs and prints one.

It does not reimplement the metric: rule_following_acc and load_model are
imported from evaluate_cp_bass (bare base models) and evaluate_cp_yinyang
(adapter models), and the key partitions come from evaluate_cp_bass.

Everything that changes how a model is REBUILT is read off its checkpoint
rather than restated on the command line, following probe_run.sh: a model
rebuilt at the wrong shape loads none of its weights and still prints a
plausible table.

  rule_view   from the k_proj width (16 = analytical, else a compiled view)
  lora_rank   from the lora_A tensor shape, if any
  rule_input  from the rule_input_encoder.* keys

Usage
-----
  python -m midi_adapter.cp_results_table \\
      --base_ckpt ckpt/cp_bass_pre_A_aligned/....ckpt \\
      --ft_ckpt   ckpt/cp_bass_ft_B_aligned/....ckpt \\
      --ckpt_dir  checkpoints_cp \\
      --runs cp_yy_frozen cp_yy_lora cp_yy_trainar cp_yy_scratch \\
             cp_yy_noinput_mlp_aux1 \\
      --n_prompt_beats 1 --n_gen_beats 23
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.evaluate_cp_bass import (
    CATEGORIES, ALL_SEEN, UNSEEN,
    load_model         as load_base_model,
    rule_following_acc as base_rule_following_acc,
)
from midi_adapter.evaluate_cp_yinyang import (
    load_model         as load_adapter_model,
    rule_following_acc as adapter_rule_following_acc,
)
from models.rasp_compiled_rule import VIEW_WIDTH


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def best_ckpt(run_dir: str) -> str | None:
    """Best checkpoint in a run directory.

    train_cp_yinyang names its series by whichever metric it monitored:
    <run>.{epoch}.{unseen_acc} when --unseen_data was given, else
    <run>.{epoch}.{val_loss}. Pick accordingly -- max for accuracy, min for
    loss -- and never fall back to last.ckpt, which carries no metric.
    """
    paths = [p for p in glob.glob(os.path.join(run_dir, '*.ckpt'))
             if os.path.basename(p) != 'last.ckpt']
    for field, better in (('unseen_acc', max), ('val_loss', min)):
        scored = []
        for p in paths:
            m = re.search(rf'{field}=(-?\d+(?:\.\d+)?)', os.path.basename(p))
            if m:
                scored.append((float(m.group(1)), p))
        if scored:
            return better(scored, key=lambda t: t[0])[1]
    return None


def inspect(ckpt_path: str) -> dict:
    """Recover from the checkpoint everything needed to rebuild the model."""
    raw = torch.load(ckpt_path, map_location='cpu')
    sd  = raw.get('state_dict', raw) if isinstance(raw, dict) else {}
    info = {'rule_mode': 'current', 'rule_view': 'key_phase',
            'lora_rank': 0, 'rule_input': 'tokens', 'enc': None,
            'enc_hidden': 64, 'n_adapters': 0}

    # n_skip was the one rebuild parameter still taken on trust from the
    # command line, and it is the one that fails silently: load_model calls
    # load_state_dict(strict=False) and discards the `unexpected` list, so
    # rebuilding an n_skip=1 checkpoint (12 adapters) at n_skip=4 (3 adapters)
    # loads the first three, reports nothing, and still prints a table.
    # n_adapters = n_global_layers / n_skip, so counting them recovers it.
    info['n_adapters'] = len({int(m.group(1)) for k in sd
                              for m in [re.search(r'yinyang_attn\.(\d+)\.', k)]
                              if m})

    for k, v in sd.items():
        if k.endswith('k_proj.weight'):
            w = v.shape[1]
            if w != 16:
                hit = [n for n, ww in VIEW_WIDTH.items() if ww == w]
                if not hit:
                    raise SystemExit(f'{ckpt_path}: k_proj width {w} is neither '
                                     f'16 nor any known view {VIEW_WIDTH}')
                info['rule_mode'], info['rule_view'] = 'compiled', hit[0]
            break
    for k, v in sd.items():
        if 'lora_A' in k:
            info['lora_rank'] = v.shape[0]
            break
    if any('rule_input_encoder.fc_out' in k for k in sd):
        info['rule_input'] = 'ar_state'
        w1 = next((v for k, v in sd.items() if k.endswith('rule_input_encoder.fc1.weight')),
                  None)
        if w1 is not None:
            info['enc'], info['enc_hidden'] = 'mlp', w1.shape[0]
        else:
            info['enc'] = 'linear'
    return info


# ---------------------------------------------------------------------------

def eval_model(model, is_adapter, keys, args, device) -> float:
    rfa = adapter_rule_following_acc if is_adapter else base_rule_following_acc
    per_key, _ = rfa(model, keys, args.n_gen_beats, args.n_trials,
                     device, args.temperature, args.n_prompt_beats)
    return float(np.mean([per_key[k] for k in keys]))


def main():
    p = argparse.ArgumentParser(description='One accuracy table across CP bass runs.')
    p.add_argument('--base_ckpt', required=True, help='pretrained base (Pretrain row)')
    p.add_argument('--ft_ckpt',   default=None, help='finetuned base, no adapter')
    p.add_argument('--ckpt_dir',  default='checkpoints_cp')
    p.add_argument('--runs',      nargs='+', default=[], help='run directory names')
    p.add_argument('--model_size',     type=int, default=1)
    p.add_argument('--adapter_rank',   type=int, default=256)
    p.add_argument('--n_skip',         type=int, default=4)
    p.add_argument('--n_prompt_beats', type=int, default=1)
    p.add_argument('--n_gen_beats',    type=int, default=23)
    p.add_argument('--n_trials',       type=int, default=1)
    p.add_argument('--temperature',    type=float, default=0.0)
    p.add_argument('--device',         type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    rows = [(lbl, keys, s) for lbl, keys, s in CATEGORIES]
    rows.append(('Overall', ALL_SEEN + UNSEEN, 'all 12 keys'))

    print('=' * 60)
    print('Loading models')
    print('=' * 60)
    print(f'  protocol : {args.n_prompt_beats}-beat prompt, '
          f'{args.n_gen_beats} generated, temperature {args.temperature}')

    entries = []            # (label, model, is_adapter)
    entries.append(('Pretrain', load_base_model(args.base_ckpt, args.model_size, device), False))
    if args.ft_ckpt:
        entries.append(('Finetune', load_base_model(args.ft_ckpt, args.model_size, device), False))

    for run in args.runs:
        d = os.path.join(args.ckpt_dir, run)
        c = best_ckpt(d)
        if c is None:
            print(f'  {run:<26} NO CHECKPOINT in {d} -- skipped')
            continue
        info = inspect(c)
        # n_global_layers for this model_size (cp_transformer.py:37)
        n_layers = [6, 12, 24, 32][args.model_size]
        n_skip = args.n_skip
        if info['n_adapters']:
            if n_layers % info['n_adapters']:
                raise SystemExit(
                    f'{c}: {info["n_adapters"]} adapters does not divide '
                    f'{n_layers} global layers -- wrong --model_size?')
            n_skip = n_layers // info['n_adapters']
            if n_skip != args.n_skip:
                print(f'  {run:<26} n_skip {args.n_skip} on the command line, but '
                      f'the checkpoint holds {info["n_adapters"]} adapters '
                      f'-> n_skip={n_skip}. Using the checkpoint.')
        print(f'  {run:<26} {os.path.basename(c)}')
        print(f'  {"":<26} rule_mode={info["rule_mode"]} view={info["rule_view"]} '
              f'lora_rank={info["lora_rank"]} rule_input={info["rule_input"]} '
              f'n_skip={n_skip}'
              + (f' enc={info["enc"]}' if info['enc'] else ''))
        model = load_adapter_model(
            args.base_ckpt, c, args.model_size, args.adapter_rank, n_skip,
            False, False, 'embedding', info['rule_mode'], device,
            rule_view       = info['rule_view'],
            lora_rank       = info['lora_rank'],
            rule_input      = info['rule_input'],
            rule_enc_type   = info['enc'] or 'mlp',
            rule_enc_hidden = info['enc_hidden'])
        entries.append((run, model, True))

    print()
    print('=' * 60)
    print('Evaluating')
    print('=' * 60)
    results = {}
    for lbl, model, is_ad in entries:
        results[lbl] = {}
        print(f'  {lbl} ...', end='  ', flush=True)
        for rlbl, keys, _ in rows:
            results[lbl][rlbl] = eval_model(model, is_ad, keys, args, device)
        print('  '.join(f'{results[lbl][r]:.3f}' for r, _, _ in rows))

    labels = [l for l, _, _ in entries]
    w = {l: max(len(l), 7) for l in labels}
    header = f"{'Key group':<20} {'Keys':<22}" + ''.join(f' {l:>{w[l]}}' for l in labels)
    sep = '-' * len(header)
    print()
    print('=' * len(header))
    print(f'CP BASS RULE-FOLLOWING ACCURACY  '
          f'({args.n_prompt_beats}-beat prompt, {args.n_gen_beats} generated)')
    print('=' * len(header))
    print(header); print(sep)
    for rlbl, _, kstr in rows:
        if rlbl == 'Overall':
            print(sep)
        line = f'{rlbl:<20} {kstr:<22}'
        for l in labels:
            line += f' {results[l][rlbl]:>{w[l]}.3f}'
        print(line)
    print('=' * len(header))


if __name__ == '__main__':
    main()
