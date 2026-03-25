"""
Evaluation: Rule-Following Accuracy on Generated Sequences

Metric
------
Given a starting value x, generate a sequence autoregressively from a
1-token prompt [x].  Rule-following accuracy is the fraction of generated
tokens that match the ground-truth rule:
    token[i] = (x + OFFSETS[i % 3]) % 12

Models compared
---------------
  1. Pretrain   – AR transformer trained on set A = {0-5}
  2. Finetune   – AR fine-tuned on set B = {2-7}, NO rule alignment
  3. Ft + Rule  – Adapter-patched model, frozen AR + rule model, fine-tuned on B

Evaluation categories (4-way partition of all 12 starting integers)
---------------------------------------------------------------------
  Pretrain-only  (A \ B) = {0, 1}       – seen only in pretraining
  Finetune-only  (B \ A) = {6, 7}       – seen only in fine-tuning
  Both           (A ∩ B) = {2, 3, 4, 5} – seen in both stages
  Neither                = {8,9,10,11}   – never seen in any training

Results are printed as a compact table and per-starter detail.
"""

from __future__ import annotations

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import (
    make_sequence, VOCAB_SIZE,
    AR_TRAIN_STARTERS, FINETUNE_STARTERS, TEST_STARTERS,
    EVAL_PRETRAIN_ONLY, EVAL_FINETUNE_ONLY, EVAL_BOTH, EVAL_NEITHER,
)
from models.transformer   import AutoregressiveTransformer
from models.adapter       import build_patched_model
from models.yinyang_model import build_yinyang_model


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _make_ar(d_model, n_layers, n_heads, n_cycles, device):
    return AutoregressiveTransformer(
        vocab_size  = VOCAB_SIZE,
        max_seq_len = n_cycles * 4 + 10,
        d_model     = d_model,
        n_layers    = n_layers,
        n_heads     = n_heads,
    ).to(device)


def load_pretrain(args, device):
    model = _make_ar(args.d_model, args.n_layers, args.n_heads, args.n_cycles, device)
    if os.path.exists(args.ar_ckpt):
        model.load_state_dict(torch.load(args.ar_ckpt, map_location=device))
        print(f"  Pretrain   : loaded {args.ar_ckpt}")
    else:
        print(f"  Pretrain   : WARNING {args.ar_ckpt} not found, using random weights")
    model.eval()
    return model


def load_finetune(args, device):
    model = _make_ar(args.d_model, args.n_layers, args.n_heads, args.n_cycles, device)
    if os.path.exists(args.ft_ckpt):
        model.load_state_dict(torch.load(args.ft_ckpt, map_location=device))
        print(f"  Finetune   : loaded {args.ft_ckpt}")
    else:
        print(f"  Finetune   : WARNING {args.ft_ckpt} not found, using random weights")
    model.eval()
    return model


def load_patched(args, device):
    model = build_patched_model(
        ar_ckpt_path   = args.ar_ckpt,
        max_seq_len    = args.n_cycles * 4 + 10,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        n_heads        = args.n_heads,
        force_fallback = args.force_fallback,
        device         = str(device),
    )
    if os.path.exists(args.adapter_ckpt):
        model.adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))
        print(f"  Ft + Rule  : loaded AR={args.ar_ckpt}, adapter={args.adapter_ckpt}")
    else:
        print(f"  Ft + Rule  : WARNING {args.adapter_ckpt} not found, using random adapter")
    model.eval()
    return model


def load_yinyang(label, ckpt_path, args, device):
    model = build_yinyang_model(
        ar_ckpt_path   = args.ar_ckpt,
        max_seq_len    = args.n_cycles * 4 + 10,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        n_heads        = args.n_heads,
        use_lora       = False,
        force_fallback = args.force_fallback,
        device         = str(device),
    )
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        yinyang_state = {k.removeprefix('yinyang_attn.'): v
                         for k, v in state.items() if k.startswith('yinyang_attn.')}
        model.yinyang_attn.load_state_dict(yinyang_state)
        print(f"  {label:<12}: loaded yinyang_attn from {ckpt_path}")
    else:
        print(f"  {label:<12}: WARNING {ckpt_path} not found")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core metric: rule-following accuracy on generated sequences
# ---------------------------------------------------------------------------

@torch.no_grad()
def rule_following_acc(model, starters, n_cycles, n_prompt, device):
    """
    For each starter x:
      - Build full ground-truth sequence of length 3*n_cycles
      - Feed first n_prompt tokens as prompt
      - Generate remaining (3*n_cycles - n_prompt) tokens
      - Compute fraction of generated tokens matching ground truth

    Returns dict {x: accuracy} and mean accuracy.
    """
    gen_len = n_cycles * 4 - n_prompt
    results = {}
    for x in starters:
        seq      = make_sequence(x, n_cycles)                               # list, length 3*n_cycles
        prompt   = torch.tensor([seq[:n_prompt]], dtype=torch.long, device=device)  # (1, n_prompt)
        expected = seq[n_prompt:]                                            # list, length gen_len

        generated_tensor = model.generate(prompt, n_new=gen_len)            # (1, n_prompt+gen_len)
        generated = generated_tensor[0, n_prompt:].cpu().tolist()

        correct = sum(g == e for g, e in zip(generated, expected))
        results[x] = correct / len(expected)

    return results, float(np.mean(list(results.values())))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Loading models")
    print("=" * 60)
    pretrain_model  = load_pretrain(args, device)
    finetune_model  = load_finetune(args, device)
    patched_model   = load_patched(args, device)
    yinyang_opt1    = load_yinyang("YY-opt1", os.path.join(args.ckpt_dir, "yinyang_opt1.pt"), args, device)
    yinyang_opt2    = load_yinyang("YY-opt2", os.path.join(args.ckpt_dir, "yinyang_opt2.pt"), args, device)
    yinyang_opt3    = load_yinyang("YY-opt3", os.path.join(args.ckpt_dir, "yinyang_opt3.pt"), args, device)

    print()
    print(f"Pretrain  set A : {AR_TRAIN_STARTERS}")
    print(f"Finetune  set B : {FINETUNE_STARTERS}")
    print(f"Eval partitions :")
    print(f"  Pretrain-only (A\\B) = {EVAL_PRETRAIN_ONLY}")
    print(f"  Finetune-only (B\\A) = {EVAL_FINETUNE_ONLY}")
    print(f"  Both          (A∩B) = {EVAL_BOTH}")
    print(f"  Neither             = {EVAL_NEITHER}")

    n_prompt = args.prompt_len

    # Evaluate all 3 models on all 4 categories
    categories = [
        ("Pretrain-only", EVAL_PRETRAIN_ONLY,  "A\\B = {0,1}"),
        ("Finetune-only", EVAL_FINETUNE_ONLY,  "B\\A = {6,7}"),
        ("Both          ", EVAL_BOTH,           "A∩B = {2,3,4,5}"),
        ("Neither       ", EVAL_NEITHER,        "{8,9,10,11}"),
    ]

    models = [
        ("Pretrain",  pretrain_model),
        ("Finetune",  finetune_model),
        ("Ft+Rule",   patched_model),
        ("YY-opt1",   yinyang_opt1),
        ("YY-opt2",   yinyang_opt2),
        ("YY-opt3",   yinyang_opt3),
    ]

    # Collect all results: results[cat_label][model_label] = (per_starter_dict, mean)
    all_results = {}
    for cat_label, starters, _ in categories:
        all_results[cat_label] = {}
        for mdl_label, mdl in models:
            res, mean = rule_following_acc(mdl, starters, args.n_cycles, n_prompt, device)
            all_results[cat_label][mdl_label] = (res, mean)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"RULE-FOLLOWING ACCURACY  (generation from {n_prompt}-token prompt)")
    print("=" * 72)

    col_w = 10
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
    row = f"{'Overall':<20} {'all 12 starters':<18}"
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
    example_starters = [
        ("Pretrain-only", EVAL_PRETRAIN_ONLY[0]),
        ("Finetune-only", EVAL_FINETUNE_ONLY[0]),
        ("Both          ", EVAL_BOTH[0]),
        ("Neither       ", EVAL_NEITHER[0]),
    ]
    show_len = 9  # tokens to show after prompt
    for cat_label, x in example_starters:
        seq      = make_sequence(x, args.n_cycles)
        prompt   = torch.tensor([seq[:n_prompt]], dtype=torch.long, device=device)
        expected = seq[n_prompt: n_prompt + show_len]

        def mark(gen): return "✓" if gen == expected else "✗"
        print(f"  [{cat_label}] x={x}  prompt={seq[:n_prompt]}")
        print(f"    Expected : {expected}")
        for mdl_label, mdl in models:
            gen = mdl.generate(prompt, show_len)[0, n_prompt:].cpu().tolist()
            print(f"    {mdl_label:<12}: {gen}  {mark(gen)}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Evaluate pretrain / finetune / ft+rule models")
    p.add_argument("--ar_ckpt",        type=str,  default="checkpoints/ar_transformer.pt",
                   help="Pretrained AR checkpoint")
    p.add_argument("--ft_ckpt",        type=str,  default="checkpoints/ar_finetuned.pt",
                   help="Fine-tuned AR checkpoint (no rule alignment)")
    p.add_argument("--adapter_ckpt",   type=str,  default="checkpoints/adapter.pt",
                   help="Adapter checkpoint for fine-tuned + rule model")
    p.add_argument("--ckpt_dir",       type=str,  default="checkpoints")
    p.add_argument("--d_model",        type=int,  default=128)
    p.add_argument("--n_layers",       type=int,  default=4)
    p.add_argument("--n_heads",        type=int,  default=4)
    p.add_argument("--n_cycles",       type=int,  default=8)
    p.add_argument("--prompt_len",     type=int,  default=1,
                   help="Number of prompt tokens (default 1 = starting value only)")
    p.add_argument("--verbose",        action="store_true",
                   help="Print per-starter accuracy breakdown")
    p.add_argument("--force_fallback", action="store_true", default=True,
                   help="Use FallbackRuleModel (default True: TracrRuleModel requires T>=3 tokens "
                        "but generation starts from T=1, so Tracr gives wrong rule_logits "
                        "for the first two steps of every sequence)")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(get_args())
