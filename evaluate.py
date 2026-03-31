from __future__ import annotations

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import (
    make_sequence,
    AR_TRAIN_STARTERS, FINETUNE_STARTERS,
    EVAL_PRETRAIN_ONLY, EVAL_FINETUNE_ONLY, EVAL_BOTH, EVAL_NEITHER,
)
from models.transformer   import AutoregressiveTransformer
from models.yinyang_model import build_yinyang_model


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _make_ar(args, device):
    from data.dataset import VOCAB_SIZE
    return AutoregressiveTransformer(
        vocab_size  = VOCAB_SIZE,
        max_seq_len = args.n_cycles * 4 + 10,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
    ).to(device)


def load_pretrain(args, device):
    model = _make_ar(args, device)
    if os.path.exists(args.ar_ckpt):
        model.load_state_dict(torch.load(args.ar_ckpt, map_location=device))
        print(f"  Pretrain   : loaded {args.ar_ckpt}")
    else:
        print(f"  Pretrain   : WARNING {args.ar_ckpt} not found, using random weights")
    model.eval()
    return model


def load_finetune(args, device):
    model = _make_ar(args, device)
    if os.path.exists(args.ft_ckpt):
        model.load_state_dict(torch.load(args.ft_ckpt, map_location=device))
        print(f"  Finetune   : loaded {args.ft_ckpt}")
    else:
        print(f"  Finetune   : WARNING {args.ft_ckpt} not found, using random weights")
    model.eval()
    return model


def load_yinyang(label, ckpt_path, args, device):
    if not os.path.exists(ckpt_path):
        print(f"  {label:<12}: WARNING {ckpt_path} not found")
        model = build_yinyang_model(
            ar_ckpt_path=args.ar_ckpt, max_seq_len=args.n_cycles * 4 + 10,
            d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
            use_lora=False, force_fallback=args.force_fallback, device=str(device),
        )
        model.eval()
        return model

    state = torch.load(ckpt_path, map_location=device)
    has_lora = any('lora_' in k for k in state)

    lora_rank = 16
    for k, v in state.items():
        if 'lora_A' in k:
            lora_rank = v.shape[0]
            break

    # Detect n_skip from number of yinyang_attn modules in checkpoint
    n_adapters = len({k.split('.')[1] for k in state if k.startswith('yinyang_attn.')})
    n_skip = args.n_layers // max(n_adapters, 1)

    model = build_yinyang_model(
        ar_ckpt_path   = args.ar_ckpt,
        max_seq_len    = args.n_cycles * 4 + 10,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        n_heads        = args.n_heads,
        n_skip         = n_skip,
        use_lora       = has_lora,
        lora_rank      = lora_rank,
        force_fallback = args.force_fallback,
        device         = str(device),
    )

    yinyang_state = {k.removeprefix('yinyang_attn.'): v
                     for k, v in state.items() if k.startswith('yinyang_attn.')}
    model.yinyang_attn.load_state_dict(yinyang_state, strict=False)

    if has_lora:
        lora_state = {k.removeprefix('ar_model.'): v
                      for k, v in state.items() if k.startswith('ar_model.')}
        model.ar_model.load_state_dict(lora_state, strict=False)
        lora_keys = [k for k in lora_state if 'lora_' in k]
        print(f"  {label:<12}: loaded yinyang_attn + LoRA (rank={lora_rank}, {len(lora_keys)} lora tensors) from {ckpt_path}")
    else:
        print(f"  {label:<12}: loaded yinyang_attn from {ckpt_path}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core metric: rule-following accuracy on generated sequences
# ---------------------------------------------------------------------------

@torch.no_grad()
def rule_following_acc(model, starters, n_cycles, n_prompt, device):
    gen_len = n_cycles * 4 - n_prompt
    results = {}
    for x in starters:
        seq      = make_sequence(x, n_cycles)
        prompt   = torch.tensor([seq[:n_prompt]], dtype=torch.long, device=device)
        expected = seq[n_prompt:]

        generated = model.generate(prompt, n_new=gen_len)[0, n_prompt:].cpu().tolist()
        results[x] = sum(g == e for g, e in zip(generated, expected)) / len(expected)

    return results, float(np.mean(list(results.values())))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Loading models")
    print("=" * 60)
    pretrain_model = load_pretrain(args, device)
    finetune_model = load_finetune(args, device)
    yinyang_skip1  = load_yinyang("skip=1", os.path.join(args.ckpt_dir, "yinyang_skip1.pt"), args, device)
    yinyang_skip2  = load_yinyang("skip=2", os.path.join(args.ckpt_dir, "yinyang_skip2.pt"), args, device)
    yinyang_skip4  = load_yinyang("skip=4", os.path.join(args.ckpt_dir, "yinyang_skip4.pt"), args, device)

    print()
    print(f"Pretrain  set A : {AR_TRAIN_STARTERS}")
    print(f"Finetune  set B : {FINETUNE_STARTERS}")
    print(f"Eval partitions :")
    print(f"  Pretrain-only (A\\B) = {EVAL_PRETRAIN_ONLY}")
    print(f"  Finetune-only (B\\A) = {EVAL_FINETUNE_ONLY}")
    print(f"  Both          (A∩B) = {EVAL_BOTH}")
    print(f"  Neither             = {EVAL_NEITHER}")

    n_prompt = args.prompt_len

    categories = [
        ("Pretrain-only", EVAL_PRETRAIN_ONLY,  "A\\B = {0,1}"),
        ("Finetune-only", EVAL_FINETUNE_ONLY,  "B\\A = {6..15}"),
        ("Both          ", EVAL_BOTH,           "A∩B = {2..5}"),
        ("Neither       ", EVAL_NEITHER,        "{17,19,20,21}"),
    ]

    models = [
        ("Pretrain",  pretrain_model),
        ("Finetune",  finetune_model),
        ("skip=1",    yinyang_skip1),
        ("skip=2",    yinyang_skip2),
        ("skip=4",    yinyang_skip4),
    ]

    all_results = {}
    for cat_label, starters, _ in categories:
        all_results[cat_label] = {}
        for mdl_label, mdl in models:
            res, mean = rule_following_acc(mdl, starters, args.n_cycles, n_prompt, device)
            all_results[cat_label][mdl_label] = (res, mean)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"RULE-FOLLOWING ACCURACY  (generation from {n_prompt}-token prompt)")
    print("=" * 72)

    col_w = 14
    mdl_labels = [m[0] for m in models]
    header = f"{'Data Split':<20} {'Starters':<18}" + "".join(f" {l:>{col_w}}" for l in mdl_labels)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for cat_label, starters, starters_str in categories:
        r = all_results[cat_label]
        row = f"{cat_label:<20} {starters_str:<18}"
        for lbl in mdl_labels:
            row += f" {r[lbl][1]:>{col_w}.3f}"
        print(row)
    print(sep)
    all_starters = EVAL_PRETRAIN_ONLY + EVAL_FINETUNE_ONLY + EVAL_BOTH + EVAL_NEITHER
    overall = {}
    for mdl_label, mdl in models:
        _, mean = rule_following_acc(mdl, all_starters, args.n_cycles, n_prompt, device)
        overall[mdl_label] = mean
    row = f"{'Overall':<20} {'all 16 starters':<18}"
    for lbl in mdl_labels:
        row += f" {overall[lbl]:>{col_w}.3f}"
    print(row)
    print("=" * len(header))

    # -----------------------------------------------------------------------
    # Per-starter detail
    # -----------------------------------------------------------------------
    if args.verbose:
        print()
        print("Per-starter breakdown")
        per_header = f"{'Starter':>8} {'Category':<18}" + "".join(f" {l:>{col_w}}" for l in mdl_labels)
        per_sep = "-" * len(per_header)
        print(per_sep)
        print(per_header)
        print(per_sep)
        for cat_label, starters, _ in categories:
            r = all_results[cat_label]
            for x in starters:
                row = f"{x:>8} {cat_label:<18}"
                for lbl in mdl_labels:
                    row += f" {r[lbl][0][x]:>{col_w}.3f}"
                print(row)
        print(per_sep)

    # -----------------------------------------------------------------------
    # Qualitative generation examples (one per category)
    # -----------------------------------------------------------------------
    print()
    print("Qualitative generation examples (prompt length =", n_prompt, ")")
    print("-" * 72)
    show_len = 9
    for cat_label, x in (
        [("Pretrain-only", EVAL_PRETRAIN_ONLY[0]),
         ("Finetune-only", EVAL_FINETUNE_ONLY[0]),
         ("Both          ", EVAL_BOTH[0])]
        + [("Neither       ", x) for x in EVAL_NEITHER]
    ):
        seq      = make_sequence(x, args.n_cycles)
        prompt   = torch.tensor([seq[:n_prompt]], dtype=torch.long, device=device)
        expected = seq[n_prompt: n_prompt + show_len]

        def mark(gen): return "✓" if gen == expected else "✗"
        print(f"  [{cat_label}] x={x}  prompt={seq[:n_prompt]}")
        print(f"    Expected : {expected}")
        for mdl_label, mdl in models:
            gen = mdl.generate(prompt, show_len)[0, n_prompt:].cpu().tolist()
            print(f"    {mdl_label:<14}: {gen}  {mark(gen)}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Evaluate pretrain / finetune / finetune+rule models")
    p.add_argument("--ar_ckpt",        type=str,  default="checkpoints/ar_transformer.pt")
    p.add_argument("--ft_ckpt",        type=str,  default="checkpoints/ar_finetuned.pt")
    p.add_argument("--ckpt_dir",       type=str,  default="checkpoints")
    p.add_argument("--d_model",        type=int,  default=128)
    p.add_argument("--n_layers",       type=int,  default=4)
    p.add_argument("--n_heads",        type=int,  default=4)
    p.add_argument("--n_cycles",       type=int,  default=8)
    p.add_argument("--prompt_len",     type=int,  default=1)
    p.add_argument("--verbose",        action="store_true")
    p.add_argument("--force_fallback", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(get_args())
