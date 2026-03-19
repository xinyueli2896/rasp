"""
Evaluation script.

Compares three models on both seen (0-7) and unseen (8-11) starting integers:
  1. AR Transformer (pretrained, no adapter)
  2. Rule Model only  (tracr or fallback)
  3. Patched Model    (AR + Rule + Adapter)

Metrics:
  - Token-level accuracy (next-token prediction)
  - Full-sequence accuracy (all tokens in generated sequence correct)
  - Per-starting-integer accuracy
"""

from __future__ import annotations

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset    import (make_sequence, AR_TRAIN_STARTERS, ADAPTER_TRAIN_STARTERS,
                              TEST_STARTERS, VOCAB_SIZE, OFFSETS, TRAIN_STARTERS)
from models.transformer import AutoregressiveTransformer
from models.rule_model   import RuleModelWrapper
from models.adapter      import build_patched_model, PatchedModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ar_model(ckpt_path, n_cycles, d_model=128, n_layers=4, n_heads=4, device="cpu"):
    model = AutoregressiveTransformer(
        vocab_size  = VOCAB_SIZE,
        max_seq_len = n_cycles * 3 + 10,
        d_model     = d_model,
        n_layers    = n_layers,
        n_heads     = n_heads,
    ).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded AR weights from {ckpt_path}")
    else:
        print("WARNING: AR checkpoint not found; using random weights.")
    model.eval()
    return model


@torch.no_grad()
def evaluate_next_token(model, starters, n_cycles, device, model_type="ar"):
    """
    Evaluate next-token prediction accuracy.

    Returns per-starter and overall accuracy.
    """
    results = {}
    for x in starters:
        seq = make_sequence(x, n_cycles)           # length = 3*n_cycles
        inp = torch.tensor([seq[:-1]], dtype=torch.long, device=device)  # (1, T-1)
        tgt = torch.tensor([seq[1:]],  dtype=torch.long, device=device)  # (1, T-1)

        if model_type == "ar":
            logits = model(inp)
        elif model_type == "rule":
            logits_np = model._rule_model.get_logits_vectorized(inp.cpu().numpy())
            logits = torch.from_numpy(logits_np).to(device)
        else:  # patched
            logits = model(inp)

        preds = logits.argmax(-1)                  # (1, T-1)
        correct = (preds == tgt).float().mean().item()
        results[x] = correct

    overall = np.mean(list(results.values()))
    return results, overall


@torch.no_grad()
def evaluate_generation(model, starters, n_cycles, n_prompt=3, device="cpu", model_type="ar"):
    """
    Evaluate full-sequence generation starting from a prompt of n_prompt tokens.

    Returns per-starter accuracy (fraction of generated tokens that are correct).
    """
    results = {}
    for x in starters:
        seq      = make_sequence(x, n_cycles)
        prompt   = torch.tensor([seq[:n_prompt]], dtype=torch.long, device=device)
        expected = seq[n_prompt:]

        if model_type == "rule":
            # Rule model is position-based: feed full context to get next tokens
            generated = list(seq[:n_prompt])
            cur = prompt
            for _ in range(len(expected)):
                logits_np = model._rule_model.get_logits_vectorized(cur.cpu().numpy())
                next_tok = int(np.argmax(logits_np[0, -1]))
                generated.append(next_tok)
                cur = torch.tensor([generated], dtype=torch.long, device=device)
        else:
            generated_tensor = model.generate(prompt, n_new=len(expected))
            generated = generated_tensor[0].cpu().tolist()

        gen_part  = generated[n_prompt:]
        correct   = sum(g == e for g, e in zip(gen_part, expected)) / len(expected)
        results[x] = correct

    overall = np.mean(list(results.values()))
    return results, overall


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cycles = args.n_cycles

    print("=" * 60)
    print("Loading models …")
    print("=" * 60)

    # 1. AR model
    ar_model = load_ar_model(args.ar_ckpt, n_cycles, args.d_model,
                              args.n_layers, args.n_heads, str(device))

    # 2. Rule model
    rule_wrapper = RuleModelWrapper(max_seq_len=n_cycles * 3 + 10,
                                    force_fallback=args.force_fallback).to(device)

    # 3. Patched model
    patched = build_patched_model(
        ar_ckpt_path   = args.ar_ckpt,
        max_seq_len    = n_cycles * 3 + 10,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        n_heads        = args.n_heads,
        force_fallback = args.force_fallback,
        device         = str(device),
    )
    if args.adapter_ckpt and os.path.exists(args.adapter_ckpt):
        patched.adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))
        print(f"Loaded adapter weights from {args.adapter_ckpt}")
    else:
        print("WARNING: adapter checkpoint not found; using random adapter.")
    patched.eval()

    print()

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    seen_starters    = AR_TRAIN_STARTERS
    adapter_starters = ADAPTER_TRAIN_STARTERS
    unseen_starters  = TEST_STARTERS

    header = f"{'Starter':>8} | {'AR':>6} | {'Rule':>6} | {'Patched':>8}"
    sep    = "-" * len(header)

    for label, starters in [
        ("AR-SEEN (0-5)",    seen_starters),
        ("ADAPTER-ONLY (6-7)", [s for s in adapter_starters if s not in seen_starters]),
        ("UNSEEN (test 8-11)", unseen_starters),
    ]:
        print(f"\n--- Next-token accuracy  [{label}] ---")
        print(header); print(sep)

        ar_res, _   = evaluate_next_token(ar_model,   starters, n_cycles, device, "ar")
        ru_res, _   = evaluate_next_token(rule_wrapper, starters, n_cycles, device, "rule")
        pa_res, _   = evaluate_next_token(patched,     starters, n_cycles, device, "patched")

        for x in starters:
            print(f"{x:>8} | {ar_res[x]:>6.3f} | {ru_res[x]:>6.3f} | {pa_res[x]:>8.3f}")
        print(sep)
        ar_mean = np.mean(list(ar_res.values()))
        ru_mean = np.mean(list(ru_res.values()))
        pa_mean = np.mean(list(pa_res.values()))
        print(f"{'MEAN':>8} | {ar_mean:>6.3f} | {ru_mean:>6.3f} | {pa_mean:>8.3f}")

    print()
    for label, starters in [
        ("AR-SEEN (0-5)",    seen_starters),
        ("ADAPTER-ONLY (6-7)", [s for s in adapter_starters if s not in seen_starters]),
        ("UNSEEN (test 8-11)", unseen_starters),
    ]:
        print(f"\n--- Generation accuracy (prompt={args.prompt_len}) [{label}] ---")
        print(header); print(sep)

        ar_res, _  = evaluate_generation(ar_model,    starters, n_cycles, args.prompt_len, str(device), "ar")
        ru_res, _  = evaluate_generation(rule_wrapper, starters, n_cycles, args.prompt_len, str(device), "rule")
        pa_res, _  = evaluate_generation(patched,      starters, n_cycles, args.prompt_len, str(device), "patched")

        for x in starters:
            print(f"{x:>8} | {ar_res[x]:>6.3f} | {ru_res[x]:>6.3f} | {pa_res[x]:>8.3f}")
        print(sep)
        ar_mean = np.mean(list(ar_res.values()))
        ru_mean = np.mean(list(ru_res.values()))
        pa_mean = np.mean(list(pa_res.values()))
        print(f"{'MEAN':>8} | {ar_mean:>6.3f} | {ru_mean:>6.3f} | {pa_mean:>8.3f}")

    # -----------------------------------------------------------------------
    # Qualitative example
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Qualitative generation example (unseen starters)")
    print("=" * 60)
    for x in TEST_STARTERS:
        seq      = make_sequence(x, n_cycles)
        prompt   = torch.tensor([seq[:args.prompt_len]], dtype=torch.long, device=device)
        expected = seq[args.prompt_len:args.prompt_len + 9]

        ar_gen = ar_model.generate(prompt, n_new=9)[0].cpu().tolist()[args.prompt_len:]
        pa_gen = patched.generate(prompt, n_new=9)[0].cpu().tolist()[args.prompt_len:]

        print(f"\n  x={x}  prompt={seq[:args.prompt_len]}")
        print(f"  Expected : {expected}")
        print(f"  AR only  : {ar_gen}  {'✓' if ar_gen == expected else '✗'}")
        print(f"  Patched  : {pa_gen}  {'✓' if pa_gen == expected else '✗'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Evaluate AR / Rule / Patched models")
    p.add_argument("--ar_ckpt",      type=str,   default="checkpoints/ar_transformer.pt")
    p.add_argument("--adapter_ckpt", type=str,   default="checkpoints/adapter.pt")
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_cycles",     type=int,   default=8)
    p.add_argument("--prompt_len",   type=int,   default=3,
                   help="Number of prompt tokens for generation eval")
    p.add_argument("--force_fallback", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(get_args())
