"""
Train the adapter that patches the rule model onto the frozen AR transformer.

Both the AR transformer and the rule model are frozen.
Only the RuleAdapter parameters are updated.

The training data is the *same subset* of starting integers (0-7) used to
pretrain the AR model.  The adapter learns a gating/blending function and
then at test time generalises to starting integers 8-11 because the rule
model is correct for all of them.
"""

from __future__ import annotations

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset  import (get_adapter_dataloaders,
                           AR_TRAIN_STARTERS, ADAPTER_TRAIN_STARTERS,
                           TEST_STARTERS, VOCAB_SIZE)
from models.adapter import build_patched_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AR pretrain starters: {AR_TRAIN_STARTERS}")
    print(f"Adapter train starters: {ADAPTER_TRAIN_STARTERS}  (includes {set(ADAPTER_TRAIN_STARTERS)-set(AR_TRAIN_STARTERS)} where AR is imperfect)")
    print(f"Test starters: {TEST_STARTERS}")

    # Data: adapter trains on ADAPTER_TRAIN_STARTERS (wider than AR pretraining)
    # Starters 6-7 are seen by adapter but NOT by AR → training signal to trust rule model
    train_loader, test_loader = get_adapter_dataloaders(
        batch_size         = args.batch_size,
        n_cycles           = args.n_cycles,
        n_seqs_per_starter = args.n_seqs_per_starter,
    )

    # Build patched model: loads frozen AR weights, builds frozen rule model, adds adapter
    model = build_patched_model(
        ar_ckpt_path   = args.ar_ckpt,
        max_seq_len    = args.n_cycles * 4 + 10,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        n_heads        = args.n_heads,
        adapter_rank   = args.adapter_rank,
        force_fallback = args.force_fallback,
        device         = str(device),
    )

    # Verify only adapter params are trainable
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params (adapter): {n_trainable:,}  |  Frozen: {n_frozen:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        # Ensure base models stay in eval / no-grad even during adapter training
        model.ar_model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)

            rule_logits = model.rule_model(inp)
            logits      = model(inp)

            # 1. Cross-entropy: predict the correct next token
            ce_loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

            # 2. KL distillation: align with the rule model (knowledge distillation)
            #    This forces the adapter to follow the rule model's predictions,
            #    enabling generalisation to unseen starting integers.
            #    Rule logits are one-hot, so this is equivalent to CE with rule targets.
            rule_probs    = torch.softmax(rule_logits, dim=-1).reshape(-1, VOCAB_SIZE)
            adapter_lprob = torch.log_softmax(logits, dim=-1).reshape(-1, VOCAB_SIZE)
            kl_loss = torch.nn.functional.kl_div(
                adapter_lprob, rule_probs, reduction='batchmean')

            loss = ce_loss + args.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            optimizer.step()

            total_loss    += loss.item() * tgt.numel()
            total_correct += (logits.argmax(-1) == tgt).sum().item()
            total_tokens  += tgt.numel()

        scheduler.step()
        train_loss = total_loss / total_tokens
        train_acc  = total_correct / total_tokens

        # --- Eval on unseen starters ---
        model.eval()
        with torch.no_grad():
            test_correct, test_tokens = 0, 0
            for inp, tgt in test_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)
                test_correct += (logits.argmax(-1) == tgt).sum().item()
                test_tokens  += tgt.numel()
        test_acc = test_correct / test_tokens

        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"test_acc(unseen)={test_acc:.4f}"
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            path = os.path.join(args.ckpt_dir, "adapter.pt")
            torch.save(model.adapter.state_dict(), path)

    print(f"\nSaved best adapter checkpoint → {os.path.join(args.ckpt_dir, 'adapter.pt')}")
    print(f"Best test accuracy on unseen starters: {best_test_acc:.4f}")

    # --- Attention analysis: what rule positions does each AR position attend to? ---
    _analyze_attn(model, test_loader, device)


def _analyze_attn(model, loader, device):
    """Print mean cross-attention weights per (query cycle pos, key cycle pos) pair.
    Averages over all heads and batch items."""
    model.eval()

    n_cyc  = 4
    adapter = model.adapter
    attn_sum   = [[0.0] * n_cyc for _ in range(n_cyc)]
    attn_count = [[0]   * n_cyc for _ in range(n_cyc)]

    with torch.no_grad():
        for inp, _ in loader:
            inp = inp.to(device)
            ar_logits, ar_hidden         = model.ar_model(inp, return_hidden=True)
            _,         rule_hidden       = model.rule_model(inp, return_hidden=True)

            B, T, _ = ar_hidden.shape
            Q = adapter.q_linear(ar_hidden)    # (B, T, embed_dim)
            K = adapter.k_linear(rule_hidden)  # (B, T, embed_dim)

            # Multi-head reshape
            Q = Q.view(B, T, adapter.n_heads, adapter.head_dim).transpose(1, 2)
            K = K.view(B, T, adapter.n_heads, adapter.head_dim).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) * adapter.scale  # (B, H, T, T)

            t_idx = torch.arange(T, device=inp.device)
            bias  = adapter.cycle_bias[t_idx[:, None] % adapter.cycle_length,
                                       t_idx[None, :] % adapter.cycle_length]
            scores = scores + bias.unsqueeze(0).unsqueeze(0)

            causal = torch.ones(T, T, device=inp.device).tril().bool()
            scores = scores.masked_fill(~causal, float('-inf'))
            attn   = F.softmax(scores, dim=-1)           # (B, H, T, T)

            # Average over batch and heads → (T, T)
            attn_mean = attn.mean(dim=(0, 1)).cpu().numpy()
            for q in range(T):
                for k in range(q + 1):
                    qc = q % n_cyc
                    kc = k % n_cyc
                    attn_sum[qc][kc]   += attn_mean[q, k]
                    attn_count[qc][kc] += 1

    print(f"\nMean cross-attention (avg over {adapter.n_heads} heads): "
          f"row=query cycle, col=key cycle")
    print("  (each row sums to ≤ 1; earlier cycles share weight with current)")
    header = "       " + "  ".join(f"key={k}" for k in range(n_cyc))
    print(header)
    for q in range(n_cyc):
        row = []
        for k in range(n_cyc):
            c = attn_count[q][k]
            row.append(f"{attn_sum[q][k]/c:.3f}" if c > 0 else "  -  ")
        print(f"  q={q}:  " + "  ".join(row))

    print(f"\nAdapter gates value: {adapter.gates.item():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train adapter over frozen AR + rule model")
    p.add_argument("--ar_ckpt",            type=str,   default="checkpoints/ar_transformer.pt")
    p.add_argument("--epochs",             type=int,   default=100)
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=3e-3)
    p.add_argument("--d_model",            type=int,   default=128)
    p.add_argument("--n_layers",           type=int,   default=4)
    p.add_argument("--n_heads",            type=int,   default=4)
    p.add_argument("--adapter_rank",       type=int,   default=32,
                   help="embed_dim for low-rank adapter (divisible by 4 heads)")
    p.add_argument("--n_cycles",           type=int,   default=8)
    p.add_argument("--n_seqs_per_starter", type=int,   default=200)
    p.add_argument("--log_every",          type=int,   default=10)
    p.add_argument("--ckpt_dir",           type=str,   default="checkpoints")
    p.add_argument("--kl_weight",          type=float, default=1.0,
                   help="Weight for KL distillation loss toward rule model")
    p.add_argument("--force_fallback",     action="store_true",
                   help="Use FallbackRuleModel even if tracr is available")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
