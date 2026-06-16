from __future__ import annotations

import os
import sys
import glob
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
    n_adapters    = len({k.split('.')[1] for k in state if k.startswith('yinyang_attn.')})
    n_skip        = args.n_layers // max(n_adapters, 1)
    # Detect bidirectional: only BidirectionalYinyangAttention has ar_to_rule
    is_bidir      = any('ar_to_rule' in k for k in state)
    # Detect encoder_injected and encoder_type from saved keys
    is_enc_inj    = any(k.startswith('rule_input_encoder.') for k in state)
    enc_type      = 'softmax' if any('token_logits' in k for k in state) else 'embedding'

    model = build_yinyang_model(
        ar_ckpt_path     = args.ar_ckpt,
        max_seq_len      = args.n_cycles * 4 + 10,
        d_model          = args.d_model,
        n_layers         = args.n_layers,
        n_heads          = args.n_heads,
        n_skip           = n_skip,
        use_lora         = has_lora,
        lora_rank        = lora_rank,
        force_fallback   = args.force_fallback,
        device           = str(device),
        bidirectional    = is_bidir,
        encoder_injected = is_enc_inj,
        encoder_type     = enc_type,
    )

    yinyang_state = {k.removeprefix('yinyang_attn.'): v
                     for k, v in state.items() if k.startswith('yinyang_attn.')}
    current = model.yinyang_attn.state_dict()
    yinyang_state = {k: v for k, v in yinyang_state.items()
                     if k in current and v.shape == current[k].shape}
    model.yinyang_attn.load_state_dict(yinyang_state, strict=False)

    # Load rule_input_encoder if present (encoder_injected checkpoints)
    enc_state = {k.removeprefix('rule_input_encoder.'): v
                 for k, v in state.items() if k.startswith('rule_input_encoder.')}
    if enc_state and hasattr(model, 'rule_input_encoder'):
        try:
            model.rule_input_encoder.load_state_dict(enc_state, strict=False)
        except RuntimeError as e:
            print(f"  [warning] encoder state shape mismatch (old checkpoint?): {e}")

    ar_keys = {k: v for k, v in state.items()
               if k.startswith('ar_model.') and 'lora_' not in k}
    has_enc = bool(enc_state)
    enc_tag = ' + encoder' if has_enc else ''
    if has_lora:
        lora_state = {k.removeprefix('ar_model.'): v
                      for k, v in state.items() if k.startswith('ar_model.')}
        model.ar_model.load_state_dict(lora_state, strict=False)
        lora_keys = [k for k in lora_state if 'lora_' in k]
        print(f"  {label:<12}: loaded yinyang_attn + LoRA (rank={lora_rank}, {len(lora_keys)} lora tensors){enc_tag} from {ckpt_path}")
    elif ar_keys:
        ar_state = {k.removeprefix('ar_model.'): v for k, v in ar_keys.items()}
        model.ar_model.load_state_dict(ar_state)
        print(f"  {label:<12}: loaded yinyang_attn + AR weights{enc_tag} from {ckpt_path}")
    else:
        print(f"  {label:<12}: loaded yinyang_attn{enc_tag} from {ckpt_path}")

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

def _load_seed_models(stem: str, args, device):
    """Load all seed checkpoints for a stem. Falls back to <stem>.pt if no seeds found."""
    seed_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, f"{stem}_seed*.pt")))
    if seed_paths:
        return [load_yinyang(f"{stem}/seed{i}", p, args, device) for i, p in enumerate(seed_paths)]
    single = os.path.join(args.ckpt_dir, f"{stem}.pt")
    return [load_yinyang(stem, single, args, device)]


def _eval_models(models_list, starters, n_cycles, n_prompt, device):
    """Return (per_starter_means, mean, std) across a list of models."""
    all_means = []
    all_per   = []
    for mdl in models_list:
        res, mean = rule_following_acc(mdl, starters, n_cycles, n_prompt, device)
        all_means.append(mean)
        all_per.append(res)
    # per-starter: average across seeds
    per = {x: float(np.mean([r[x] for r in all_per])) for x in starters}
    return per, float(np.mean(all_means)), float(np.std(all_means))


def _fmt(mean, std, col_w, is_multi):
    """Format a cell as 'mean±std' (multi-seed) or 'mean' (single)."""
    if is_multi:
        s = f"{mean:.3f}±{std:.3f}"
        return s.rjust(col_w)
    return f"{mean:{col_w}.3f}"


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Loading models")
    print("=" * 60)
    pretrain_model = load_pretrain(args, device)
    finetune_model = load_finetune(args, device)

    default_stems = ["yinyang_skip1", "yinyang_skip2", "yinyang_skip4"]
    stems = args.adapter_stems if args.adapter_stems else default_stems
    stem_models = [(s, _load_seed_models(s, args, device)) for s in stems]

    print()
    print(f"Pretrain  set A : {AR_TRAIN_STARTERS}")
    print(f"Finetune  set B : {FINETUNE_STARTERS}")
    print(f"Eval partitions :")
    print(f"  Pretrain-only (A\\B) = {EVAL_PRETRAIN_ONLY}")
    print(f"  Finetune-only (B\\A) = {EVAL_FINETUNE_ONLY}")
    print(f"  Both          (A∩B) = {EVAL_BOTH}")
    print(f"  Neither             = {EVAL_NEITHER}")
    for s, mdls in stem_models:
        print(f"  {s}: {len(mdls)} seed(s)")

    n_prompt = args.prompt_len

    categories = [
        ("Pretrain-only", EVAL_PRETRAIN_ONLY,  "A\\B = {0,1}"),
        ("Finetune-only", EVAL_FINETUNE_ONLY,  "B\\A = {6..15}"),
        ("Both          ", EVAL_BOTH,           "A∩B = {2..5}"),
        ("Neither       ", EVAL_NEITHER,        "{17,19,20,21}"),
    ]

    # Single-model entries (Pretrain, Finetune) + multi-seed adapter entries
    single_models = [
        ("Pretrain", [pretrain_model]),
        ("Finetune", [finetune_model]),
    ]
    adapter_models = [(s, mdls, len(mdls) > 1) for s, mdls in stem_models]

    # Evaluate all
    # results[label] = {cat_label: (per_starter, mean, std)}
    results = {}
    all_starters = EVAL_PRETRAIN_ONLY + EVAL_FINETUNE_ONLY + EVAL_BOTH + EVAL_NEITHER
    for lbl, mdls in single_models:
        results[lbl] = {}
        for cat_label, starters, _ in categories:
            per, mean, std = _eval_models(mdls, starters, args.n_cycles, n_prompt, device)
            results[lbl][cat_label] = (per, mean, std)
        per, mean, std = _eval_models(mdls, all_starters, args.n_cycles, n_prompt, device)
        results[lbl]["Overall"] = (per, mean, std)
    for lbl, mdls, _ in adapter_models:
        results[lbl] = {}
        for cat_label, starters, _ in categories:
            per, mean, std = _eval_models(mdls, starters, args.n_cycles, n_prompt, device)
            results[lbl][cat_label] = (per, mean, std)
        per, mean, std = _eval_models(mdls, all_starters, args.n_cycles, n_prompt, device)
        results[lbl]["Overall"] = (per, mean, std)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print(f"RULE-FOLLOWING ACCURACY  (generation from {n_prompt}-token prompt)")
    print("=" * 80)

    col_w     = 14   # single-value columns
    col_w_ms  = 16   # mean±std columns (wider)
    all_labels = [lbl for lbl, _ in single_models] + [lbl for lbl, _, _ in adapter_models]
    is_multi   = {lbl: False for lbl, _ in single_models}
    is_multi.update({lbl: multi for lbl, _, multi in adapter_models})

    header = f"{'Data Split':<20} {'Starters':<18}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        header += f" {lbl:>{w}}"
    sep = "-" * len(header)

    print(header)
    print(sep)
    for cat_label, _, starters_str in categories:
        row = f"{cat_label:<20} {starters_str:<18}"
        for lbl in all_labels:
            w = col_w_ms if is_multi[lbl] else col_w
            _, mean, std = results[lbl][cat_label]
            row += " " + _fmt(mean, std, w, is_multi[lbl])
        print(row)
    print(sep)
    row = f"{'Overall':<20} {'all 16 starters':<18}"
    for lbl in all_labels:
        w = col_w_ms if is_multi[lbl] else col_w
        _, mean, std = results[lbl]["Overall"]
        row += " " + _fmt(mean, std, w, is_multi[lbl])
    print(row)
    print("=" * len(header))

    # -----------------------------------------------------------------------
    # Per-starter detail (uses seed-averaged per-starter scores)
    # -----------------------------------------------------------------------
    if args.verbose:
        print()
        print("Per-starter breakdown")
        per_header = f"{'Starter':>8} {'Category':<18}"
        for lbl in all_labels:
            w = col_w_ms if is_multi[lbl] else col_w
            per_header += f" {lbl:>{w}}"
        per_sep = "-" * len(per_header)
        print(per_sep)
        print(per_header)
        print(per_sep)
        for cat_label, starters, _ in categories:
            for x in starters:
                row = f"{x:>8} {cat_label:<18}"
                for lbl in all_labels:
                    w = col_w_ms if is_multi[lbl] else col_w
                    per, _, _ = results[lbl][cat_label]
                    row += f" {per[x]:{w}.3f}"
                print(row)
        print(per_sep)

    # -----------------------------------------------------------------------
    # Qualitative generation examples (seed 0 for adapters)
    # -----------------------------------------------------------------------
    print()
    print("Qualitative generation examples (prompt length =", n_prompt, ")")
    print("-" * 72)
    show_len = 9
    qual_models = (
        [("Pretrain", pretrain_model), ("Finetune", finetune_model)]
        + [(s, mdls[0]) for s, mdls in stem_models]
    )
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
        for mdl_label, mdl in qual_models:
            gen = mdl.generate(prompt, show_len)[0, n_prompt:].cpu().tolist()
            print(f"    {mdl_label:<14}: {gen}  {mark(gen)}")
        print()


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def run_multiseed_evaluation(stem: str, args, device):
    pattern = os.path.join(args.ckpt_dir, f"{stem}_seed*.pt")
    ckpt_paths = sorted(glob.glob(pattern))
    if not ckpt_paths:
        print(f"  No seed checkpoints found matching: {pattern}")
        return

    all_starters = EVAL_PRETRAIN_ONLY + EVAL_FINETUNE_ONLY + EVAL_BOTH + EVAL_NEITHER
    categories = [
        ("Pretrain-only", EVAL_PRETRAIN_ONLY),
        ("Finetune-only", EVAL_FINETUNE_ONLY),
        ("Both          ", EVAL_BOTH),
        ("Neither       ", EVAL_NEITHER),
        ("Overall       ", all_starters),
    ]

    # acc_per_cat[cat_label] = list of per-seed mean accuracies
    acc_per_cat = {cat: [] for cat, _ in categories}
    seed_names  = []

    for ckpt_path in ckpt_paths:
        seed_name = os.path.basename(ckpt_path).replace('.pt', '')
        seed_names.append(seed_name)
        print(f"  Evaluating {seed_name} ...", end='  ', flush=True)
        model = load_yinyang(seed_name, ckpt_path, args, device)
        model.eval()
        for cat_label, starters in categories:
            _, mean = rule_following_acc(model, starters, args.n_cycles, args.prompt_len, device)
            acc_per_cat[cat_label].append(mean)
        print("  ".join(f"{acc_per_cat[cat][-1]:.3f}" for cat, _ in categories))

    col_w = 16
    print()
    print(f"Multi-seed summary: {stem}  ({len(ckpt_paths)} seeds)")
    header = f"  {'':25}" + "".join(f" {c[0]:>{col_w}}" for c in categories)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(seed_names):
        row = f"  {name:<25}" + "".join(f" {acc_per_cat[c][i]:>{col_w}.4f}" for c, _ in categories)
        print(row)
    print("-" * len(header))
    for stat_label, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        row = f"  {stat_label:<25}" + "".join(f" {fn(acc_per_cat[c]):>{col_w}.4f}" for c, _ in categories)
        print(row)
    print("-" * len(header))


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
    p.add_argument("--seed_stems",     type=str,  nargs="*", default=[],
                   help="Checkpoint stems to evaluate across seeds, e.g. --seed_stems yinyang_skip1 yinyang_skip2")
    p.add_argument("--adapter_stems",  type=str,  nargs="*", default=None,
                   help="Override adapter stems in main table (default: yinyang_skip1 yinyang_skip2 yinyang_skip4)")
    p.add_argument("--inspect_encoder", action="store_true",
                   help="Print learned token mapping table for encoder_injected checkpoints")
    return p.parse_args()


def inspect_encoder(model, label: str):
    """Print the learned token→token mapping for encoder_injected models."""
    enc = getattr(model, 'rule_input_encoder', None)
    if enc is None or not hasattr(enc, 'token_logits'):
        print(f"  [{label}] no softmax encoder found")
        return

    with torch.no_grad():
        W = enc.token_logits.cpu() if isinstance(enc.token_logits, torch.Tensor) \
            else enc.token_logits.weight.cpu()

    if W.dim() == 2:
        # Legacy: position-agnostic (V, V)
        _print_encoder_table(W.unsqueeze(0), label, pos_names=["(all)"])
    else:
        # Position-aware: (4, V, V)
        _print_encoder_table(W, label, pos_names=[f"pos%4={p}" for p in range(4)])


def _print_encoder_table(W: torch.Tensor, label: str, pos_names: list):
    """W: (n_pos, V, V)"""
    print(f"\nEncoder mapping [{label}]")
    for p, name in enumerate(pos_names):
        probs  = torch.softmax(W[p], dim=-1)
        mapped = probs.argmax(dim=-1).tolist()
        V = W.shape[1]
        print(f"\n  {name}:")
        print(f"  {'in':>4}  {'→out':>5}  {'conf':>6}  {'top-3 (token:prob)':}")
        print(f"  {'-'*50}")
        for i in range(V):
            top3 = probs[i].topk(3)
            top3_str = "  ".join(f"{idx.item()}:{p2.item():.2f}"
                                  for idx, p2 in zip(top3.indices, top3.values))
            conf = probs[i, mapped[i]].item()
            print(f"  {i:>4}  {mapped[i]:>5}  {conf:>6.3f}  {top3_str}")


if __name__ == "__main__":
    args = get_args()
    run_evaluation(args)
    if args.inspect_encoder:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_stems = ["yinyang_skip1", "yinyang_skip2", "yinyang_skip4"]
        stems = args.adapter_stems if args.adapter_stems else default_stems
        print("\n" + "=" * 60)
        print("ENCODER INSPECTION")
        print("=" * 60)
        for stem in stems:
            seed_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, f"{stem}_seed*.pt")))
            paths = seed_paths if seed_paths else [os.path.join(args.ckpt_dir, f"{stem}.pt")]
            for i, path in enumerate(paths):
                if not os.path.exists(path):
                    continue
                label = f"{stem}/seed{i}" if seed_paths else stem
                mdl = load_yinyang(label, path, args, device)
                inspect_encoder(mdl, label)
    for stem in args.seed_stems:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print()
        print("=" * 72)
        print(f"MULTI-SEED EVALUATION: {stem}")
        print("=" * 72)
        run_multiseed_evaluation(stem, args, device)
