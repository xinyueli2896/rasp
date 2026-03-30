"""
main.py  –  end-to-end pipeline runner

Usage:
    python main.py                    # run all stages
    python main.py --stage pretrain   # pretrain AR transformer only
    python main.py --stage finetune   # fine-tune AR on set B (no rule)
    python main.py --stage yinyang    # train Yin-Yang adapter (frozen AR + rule)
    python main.py --stage eval       # evaluate all models

Flags are forwarded to the individual scripts; most defaults are set there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os


STAGES = ["pretrain", "finetune", "yinyang", "eval"]


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
    p = argparse.ArgumentParser(description="Modulo-sequence Yin-Yang pipeline")
    p.add_argument("--stage",            type=str, choices=STAGES + ["all"], default="all")
    p.add_argument("--epochs_pretrain",  type=int, default=200)
    p.add_argument("--epochs_finetune",  type=int, default=200)
    p.add_argument("--epochs_yinyang",   type=int, default=100)
    p.add_argument("--n_skip",           type=int, default=1,
                   help="Yin-Yang injection interval (1=every layer, 2=every 2, 4=last only)")
    p.add_argument("--n_cycles",         type=int, default=8)
    p.add_argument("--d_model",          type=int, default=128)
    p.add_argument("--n_layers",         type=int, default=4)
    p.add_argument("--n_heads",          type=int, default=4)
    p.add_argument("--force_fallback",   action="store_true")
    p.add_argument("--ckpt_dir",         type=str, default="checkpoints")
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
        elif stage == "finetune":
            run_stage("training/finetune.py",
                      shared + ["--epochs", str(args.epochs_finetune),
                                "--ar_ckpt", os.path.join(args.ckpt_dir, "ar_transformer.pt")])
        elif stage == "yinyang":
            run_stage("training/train_yinyang.py",
                      shared + ["--epochs",    str(args.epochs_yinyang),
                                "--n_skip",    str(args.n_skip),
                                "--no_lora",
                                "--ckpt_name", f"yinyang_skip{args.n_skip}",
                                "--ar_ckpt",   os.path.join(args.ckpt_dir, "ar_transformer.pt")])
        elif stage == "eval":
            run_stage("evaluate.py",
                      shared + ["--ar_ckpt", os.path.join(args.ckpt_dir, "ar_transformer.pt"),
                                "--ft_ckpt",  os.path.join(args.ckpt_dir, "ar_finetuned.pt"),
                                "--verbose"])


if __name__ == "__main__":
    main()
