"""
main.py  –  end-to-end pipeline runner

Usage:
    python main.py                    # run all stages
    python main.py --stage pretrain   # pretrain AR transformer only
    python main.py --stage adapter    # train adapter only (AR ckpt must exist)
    python main.py --stage eval       # evaluate all models

Flags are forwarded to the individual scripts; most defaults are set there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os


STAGES = ["pretrain", "adapter", "eval"]


def run_stage(script: str, extra_args: list[str]):
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"Stage failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(description="Modulo-sequence adapter pipeline")
    p.add_argument("--stage",          type=str, choices=STAGES + ["all"], default="all")
    p.add_argument("--epochs_pretrain",type=int, default=200)
    p.add_argument("--epochs_adapter", type=int, default=100)
    p.add_argument("--n_cycles",       type=int, default=8)
    p.add_argument("--d_model",        type=int, default=128)
    p.add_argument("--n_layers",       type=int, default=4)
    p.add_argument("--n_heads",        type=int, default=4)
    p.add_argument("--force_fallback", action="store_true",
                   help="Use FallbackRuleModel (pure NumPy) even if tracr is available")
    p.add_argument("--ckpt_dir",       type=str, default="checkpoints")
    args = p.parse_args()

    shared = [
        "--n_cycles",  str(args.n_cycles),
        "--d_model",   str(args.d_model),
        "--n_layers",  str(args.n_layers),
        "--n_heads",   str(args.n_heads),
        "--ckpt_dir",  args.ckpt_dir,
    ]
    if args.force_fallback:
        shared.append("--force_fallback")

    stages_to_run = STAGES if args.stage == "all" else [args.stage]

    for stage in stages_to_run:
        if stage == "pretrain":
            run_stage("training/pretrain.py",
                      shared + ["--epochs", str(args.epochs_pretrain)])
        elif stage == "adapter":
            run_stage("training/train_adapter.py",
                      shared + [
                          "--epochs",   str(args.epochs_adapter),
                          "--ar_ckpt",  os.path.join(args.ckpt_dir, "ar_transformer.pt"),
                      ])
        elif stage == "eval":
            run_stage("evaluate.py",
                      shared + [
                          "--ar_ckpt",      os.path.join(args.ckpt_dir, "ar_transformer.pt"),
                          "--adapter_ckpt", os.path.join(args.ckpt_dir, "adapter.pt"),
                      ])


if __name__ == "__main__":
    main()
