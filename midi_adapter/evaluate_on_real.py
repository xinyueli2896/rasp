"""
Evaluate a trained CP Yinyang adapter on REAL windows sampled from the
extracted-and-transposed .pt datasets built by `extract_orchestrated.py`.

For each window in the dataset:
  1. Take the first --n_prompt_beats subbeats as the prompt (default 16 = 1 bar).
  2. Let the adapter generate autoregressively out to the window's full length
     (n_prompt_beats + n_gen_beats subbeats).
  3. Compare the generated content against the expected I-IV-V-I chord sequence
     in the window's target key (which is stamped into the .txt sidecar file
     alongside the .pt tensor by extract_orchestrated).

Metrics per key:
  * bass-note accuracy  — voice 0's pitch class == expected root
  * chord-coverage acc  — expected {root, root+4, root+7} ⊆ set of pitch
                          classes across all voices at that subbeat

Usage
-----
    python -m midi_adapter.evaluate_on_real \\
        --base_ckpt    /path/to/pretrain.ckpt \\
        --adapter_ckpt /path/to/cp_yinyang_chord_real.pt \\
        --seen_data    /l/users/xinyue.li/data/pop909_orch_val_seenkeys.pt \\
        --unseen_data  /l/users/xinyue.li/data/pop909_orch_val_unseenkeys.pt \\
        --approach chord --encoder_injected --n_skip 1 --chords_per_bar 2 \\
        --n_prompt_beats 16 --temperature 0
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.evaluate_cp_yinyang import load_model
from midi_adapter.generate_synthetic_bass import SUBBEATS_PER_BAR, OFFSETS
from midi_adapter.filter_nottingham import ROOT_NAMES


_KEY_STR_TO_PC = {name: i for i, name in enumerate(ROOT_NAMES)}
_KEY_RE = re.compile(r'_key([A-G]#?)_|#key([A-G]#?)$')

_MAJOR_INTERVALS = (0, 4, 7)


def _parse_key(line: str) -> int | None:
    """Extract the target-key pitch class from a .txt sidecar line."""
    m = _KEY_RE.search(line)
    if not m:
        return None
    return _KEY_STR_TO_PC.get(m.group(1) or m.group(2))


def _expected_root(key: int, sb: int, chords_per_bar: int, phase: int = 0) -> int:
    sub_per_chord = SUBBEATS_PER_BAR // chords_per_bar
    return (key + OFFSETS[(sb // sub_per_chord + phase) % 4]) % 12


def _load_windows(pt_path: str, window_len: int) -> tuple[torch.Tensor, list[int]]:
    """Load a dataset produced by extract_orchestrated.

    Returns (windows: (N, window_len, subseq), keys: list of target-key pitch
    classes parsed from the .txt sidecar)."""
    data = torch.load(pt_path, weights_only=True)   # (N * window_len, subseq)
    if data.dim() != 2:
        raise ValueError(f'Unexpected data shape {data.shape}')
    n_rows, subseq = data.shape
    if n_rows % window_len != 0:
        raise ValueError(f'{pt_path} has {n_rows} rows, not divisible by window_len={window_len}')
    n_windows = n_rows // window_len
    windows = data.view(n_windows, window_len, subseq)

    txt_path = pt_path[:-3] + '.txt'
    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) != n_windows:
        print(f'  WARN: {txt_path} has {len(lines)} entries but data has {n_windows} windows')
    keys: list[int] = []
    for i in range(n_windows):
        k = _parse_key(lines[i]) if i < len(lines) else None
        keys.append(k if k is not None else -1)
    return windows, keys


@torch.no_grad()
def evaluate_dataset(model, windows: torch.Tensor, keys: list[int],
                     n_prompt_beats: int, n_gen_beats: int,
                     temperature: float, chords_per_bar: int,
                     device: torch.device, batch_size: int = 8,
                     max_windows: int | None = None
                     ) -> dict[int, dict[str, float]]:
    """Generate continuations for every window and score rule following.
    Returns per-key stats dict: {key: {'bass_acc': ..., 'chord_cov': ..., 'n': ...}}."""
    n_windows = len(windows) if max_windows is None else min(max_windows, len(windows))
    total_beats  = n_prompt_beats + n_gen_beats
    per_key: dict[int, list[tuple[float, float]]] = {}

    tokenizer = model.base.tokenizer
    S = model.base.hidden_size  # unused but keeps the tensor shape check honest

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = windows[start:end].to(device).long()
        batch_keys = [keys[i] for i in range(start, end)]

        pitch_shift = torch.zeros(end - start, dtype=torch.long, device=device)
        x_proc = model.base.preprocess(batch, pitch_shift)
        # global_sampling → local_encode → x.view(-1, ...), which needs a
        # contiguous tensor. The slice below is a non-contiguous view.
        prompt = x_proc[:, :n_prompt_beats].contiguous()

        sampled = model.global_sampling(prompt, max_seq_len=total_beats,
                                         temperature=temperature)

        # sampled[t] is (batch_size, subseq_len) preprocessed token for beat t.
        for local_i, key in enumerate(batch_keys):
            if key < 0:
                continue
            bass_ok = 0
            cov_ok  = 0
            gen_slots = range(n_prompt_beats, total_beats)
            for t in gen_slots:
                y_t = sampled[t][local_i]           # (subseq_len,)
                # Slot format after preprocess: (prog, pitch+dur*128) pairs.
                pcs = set()
                bass_pc = None
                for v in range(0, y_t.shape[0], 2):
                    a = int(y_t[v].item())
                    if a == tokenizer.eos_token or a == tokenizer.pad_token:
                        break
                    b = int(y_t[v + 1].item()) if v + 1 < y_t.shape[0] else tokenizer.pad_token
                    if b == tokenizer.eos_token or b == tokenizer.pad_token or b < 128:
                        continue
                    pc = (b % 128) % 12
                    if bass_pc is None:
                        bass_pc = pc
                    pcs.add(pc)
                expected_root = _expected_root(key, t, chords_per_bar)
                if bass_pc == expected_root:
                    bass_ok += 1
                expected_pcs = {(expected_root + i) % 12 for i in _MAJOR_INTERVALS}
                if expected_pcs.issubset(pcs):
                    cov_ok += 1
            per_key.setdefault(key, []).append(
                (bass_ok / n_gen_beats, cov_ok / n_gen_beats))

    stats: dict[int, dict[str, float]] = {}
    for k, results in per_key.items():
        bass = np.array([r[0] for r in results])
        cov  = np.array([r[1] for r in results])
        stats[k] = {
            'bass_acc':  float(bass.mean()),
            'chord_cov': float(cov.mean()),
            'n':         len(results),
        }
    return stats


def _print_stats_table(title: str, stats: dict[int, dict[str, float]]) -> None:
    print(f'\n── {title} ──')
    print(f'  {"key":<4}  {"n":>4}  {"bass_acc":>10}  {"chord_cov":>10}')
    means = {'bass': [], 'cov': []}
    for k in sorted(stats):
        s = stats[k]
        print(f'  {ROOT_NAMES[k]:<4}  {s["n"]:>4}  {s["bass_acc"]:>10.3f}  {s["chord_cov"]:>10.3f}')
        means['bass'].append(s['bass_acc'])
        means['cov'].append(s['chord_cov'])
    if means['bass']:
        print(f'  {"MEAN":<4}  {"":>4}  {np.mean(means["bass"]):>10.3f}  '
              f'{np.mean(means["cov"]):>10.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt',    default=None,
                   help='Base CP transformer ckpt (only needed for adapter-only .pt)')
    p.add_argument('--adapter_ckpt', required=True)
    p.add_argument('--seen_data',    type=str, default=None,
                   help='.pt dataset of val windows in seen keys')
    p.add_argument('--unseen_data',  type=str, default=None,
                   help='.pt dataset of val windows in unseen keys')
    p.add_argument('--n_prompt_beats', type=int, default=16,
                   help='Subbeats of prompt (default 16 = 1 bar at beat_div=4)')
    p.add_argument('--window_len',   type=int, default=64,
                   help='Length of each window in the dataset (default 64 = 4 bars).')
    p.add_argument('--n_gen_beats',  type=int, default=0,
                   help='Subbeats to generate. Default 0 → window_len - n_prompt_beats.')
    p.add_argument('--temperature',  type=float, default=0.0)
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--max_windows',  type=int, default=0,
                   help='Cap windows per dataset (0 = all)')
    p.add_argument('--model_size',   type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip',       type=int, default=1)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--encoder_injected', action='store_true')
    p.add_argument('--encoder_type', type=str, default='embedding',
                   choices=['embedding', 'token_mlp'])
    p.add_argument('--rule_mode',    type=str, default='current')
    p.add_argument('--approach',     type=str, default='chord', choices=['bass', 'chord'])
    p.add_argument('--chords_per_bar', type=int, default=2, choices=[1, 2, 4])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gen = args.n_gen_beats or (args.window_len - args.n_prompt_beats)
    max_windows = args.max_windows or None

    print('Loading model ...')
    model = load_model(args.base_ckpt, args.adapter_ckpt,
                       args.model_size, args.adapter_rank, args.n_skip,
                       args.bidirectional, args.encoder_injected,
                       args.encoder_type, args.rule_mode, args.approach, device)
    model.eval()

    for label, path in (('SEEN keys', args.seen_data),
                         ('UNSEEN keys', args.unseen_data)):
        if path is None:
            continue
        print(f'\nLoading {label}: {path}')
        windows, keys = _load_windows(path, args.window_len)
        print(f'  {len(windows)} windows')

        stats = evaluate_dataset(
            model, windows, keys,
            n_prompt_beats=args.n_prompt_beats,
            n_gen_beats   =n_gen,
            temperature   =args.temperature,
            chords_per_bar=args.chords_per_bar,
            device        =device,
            batch_size    =args.batch_size,
            max_windows   =max_windows,
        )
        _print_stats_table(f'{label}  (prompt={args.n_prompt_beats}, gen={n_gen}, '
                            f'T={args.temperature})', stats)


if __name__ == '__main__':
    main()
