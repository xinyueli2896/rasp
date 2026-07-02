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
from midi_adapter.infer_cp_yinyang import decode_output


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


def _load_windows(pt_path: str, window_len: int
                   ) -> tuple[torch.Tensor, list[int], torch.Tensor | None]:
    """Load a dataset produced by extract_orchestrated.

    Returns:
      windows: (N, window_len, subseq) uint8
      keys:    list of target-key pitch classes (parsed from .txt sidecar
                OR loaded from .keys.pt if present)
      chord_seq: (N, N_chords) long — loaded from .chord_seq.pt if present, else None
    """
    data = torch.load(pt_path, weights_only=True)   # (N * window_len, subseq)
    if data.dim() != 2:
        raise ValueError(f'Unexpected data shape {data.shape}')
    n_rows, subseq = data.shape
    if n_rows % window_len != 0:
        raise ValueError(f'{pt_path} has {n_rows} rows, not divisible by window_len={window_len}')
    n_windows = n_rows // window_len
    windows = data.view(n_windows, window_len, subseq)

    # Keys: prefer .keys.pt over parsing the .txt.
    keys_path = pt_path[:-3] + '.keys.pt'
    if os.path.exists(keys_path):
        keys = torch.load(keys_path, weights_only=True).long().tolist()
    else:
        txt_path = pt_path[:-3] + '.txt'
        with open(txt_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) != n_windows:
            print(f'  WARN: {txt_path} has {len(lines)} entries but data has {n_windows} windows')
        keys = []
        for i in range(n_windows):
            k = _parse_key(lines[i]) if i < len(lines) else None
            keys.append(k if k is not None else -1)

    # Optional paired chord_seq — required in chord-seq inference mode.
    cs_path = pt_path[:-3] + '.chord_seq.pt'
    chord_seq = None
    if os.path.exists(cs_path):
        chord_seq = torch.load(cs_path, weights_only=True).long()

    return windows, keys, chord_seq


def _detect_root_per_halfbar(sampled, local_i: int, total_beats: int,
                              sub_per_chord: int, tokenizer) -> list[int]:
    """Walk the full sampled sequence for one sample and detect the chord root
    per half-bar (= per chord slot at chords_per_bar=2). Uses the same
    chromagram + major-triad matcher as the filter."""
    n_halfbars = total_beats // sub_per_chord
    roots: list[int] = []
    for h in range(n_halfbars):
        chroma = np.zeros(12, dtype=np.float32)
        for local_sb in range(sub_per_chord):
            t   = h * sub_per_chord + local_sb
            y_t = sampled[t][local_i]
            S   = y_t.shape[0]
            for v in range(0, S, 2):
                a = int(y_t[v].item())
                if a == tokenizer.eos_token or a == tokenizer.pad_token:
                    break
                if v + 1 >= S:
                    break
                b = int(y_t[v + 1].item())
                if b == tokenizer.eos_token or b == tokenizer.pad_token or b < 128:
                    continue
                chroma[(b % 128) % 12] += 1.0
        if chroma.sum() < 1e-3:
            roots.append(-1)
            continue
        chroma /= chroma.sum()
        best_r, best_s = 0, -1.0
        for r in range(12):
            s = chroma[r] + chroma[(r + 4) % 12] + chroma[(r + 7) % 12]
            if s > best_s:
                best_s, best_r = s, r
        roots.append(best_r)
    return roots


def _print_demo(key: int, sample_i: int, gen_roots: list[int],
                prompt_halfbars: int, chords_per_bar: int) -> None:
    n_halfbars = len(gen_roots)
    expected = [(key + OFFSETS[h % 4]) % 12 for h in range(n_halfbars)]
    hb_col   = '  '.join(f'{h:>3}' for h in range(n_halfbars))
    zone     = '  '.join(f'{("P" if h < prompt_halfbars else "G"):>3}' for h in range(n_halfbars))
    exp_col  = '  '.join(f'{ROOT_NAMES[r]:>3}' for r in expected)
    got_col  = '  '.join(f'{(ROOT_NAMES[r] if r >= 0 else "-"):>3}' for r in gen_roots)
    marks    = []
    for h, (g, e) in enumerate(zip(gen_roots, expected)):
        if h < prompt_halfbars:
            marks.append('·')
        else:
            marks.append('✓' if g == e else '✗')
    mark_col = '  '.join(f'{m:>3}' for m in marks)
    # Rule-following rate over generated half-bars only.
    gen_slice = list(zip(gen_roots[prompt_halfbars:], expected[prompt_halfbars:]))
    acc = sum(1 for g, e in gen_slice if g == e) / max(len(gen_slice), 1)
    print(f'  Key {ROOT_NAMES[key]}, sample {sample_i}:  gen half-bar acc = {acc:.3f}')
    print(f'    half-bar : {hb_col}')
    print(f'    zone     : {zone}   (P = prompt half-bars, G = generated)')
    print(f'    expected : {exp_col}')
    print(f'    detected : {got_col}')
    print(f'    match    : {mark_col}')


@torch.no_grad()
def evaluate_dataset(model, windows: torch.Tensor, keys: list[int],
                     n_prompt_beats: int, n_gen_beats: int,
                     temperature: float, chords_per_bar: int,
                     device: torch.device, batch_size: int = 8,
                     max_windows: int | None = None,
                     chord_seq: torch.Tensor | None = None,
                     save_midi_dir: str | None = None,
                     midi_prefix:   str = 'seen',
                     n_demos_per_key: int = 0,
                     save_n_per_key: int = 0,
                     ) -> dict[int, dict[str, float]]:
    """Generate continuations for every window and score rule following.
    Returns per-key stats dict: {key: {'bass_acc': ..., 'chord_cov': ..., 'n': ...}}."""
    n_windows = len(windows) if max_windows is None else min(max_windows, len(windows))
    total_beats  = n_prompt_beats + n_gen_beats
    per_key: dict[int, list[tuple[float, float]]] = {}

    tokenizer     = model.base.tokenizer
    sub_per_chord = SUBBEATS_PER_BAR // chords_per_bar
    prompt_halfbars = n_prompt_beats // sub_per_chord
    demos_shown:   dict[int, int] = {}
    saved_per_key: dict[int, int] = {}

    if save_midi_dir is not None:
        os.makedirs(save_midi_dir, exist_ok=True)

    if n_demos_per_key > 0:
        print(f'\n  Qualitative demos (up to {n_demos_per_key} per key):')

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = windows[start:end].to(device).long()
        batch_keys = [keys[i] for i in range(start, end)]

        pitch_shift = torch.zeros(end - start, dtype=torch.long, device=device)
        x_proc = model.base.preprocess(batch, pitch_shift)
        # global_sampling → local_encode → x.view(-1, ...), which needs a
        # contiguous tensor. The slice below is a non-contiguous view.
        prompt = x_proc[:, :n_prompt_beats].contiguous()

        batch_chord_seq = None
        if chord_seq is not None:
            batch_chord_seq = chord_seq[start:end].to(device)

        sampled = model.global_sampling(prompt, max_seq_len=total_beats,
                                         temperature=temperature,
                                         chord_seq=batch_chord_seq)

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

            # PRIMARY metric — same chromagram chord-root detection as the
            # filter: detect one root per half-bar of the generation and count
            # matches against the expected I-IV-V-I sequence.
            gen_roots = _detect_root_per_halfbar(
                sampled, local_i, total_beats, sub_per_chord, tokenizer)
            n_hb_gen  = 0
            n_hb_ok   = 0
            for h in range(prompt_halfbars, len(gen_roots)):
                exp = (key + OFFSETS[h % 4]) % 12
                n_hb_gen += 1
                if gen_roots[h] == exp:
                    n_hb_ok += 1
            halfbar_acc = n_hb_ok / max(n_hb_gen, 1)

            per_key.setdefault(key, []).append(
                (halfbar_acc, bass_ok / n_gen_beats, cov_ok / n_gen_beats))

            if n_demos_per_key > 0 and demos_shown.get(key, 0) < n_demos_per_key:
                demos_shown[key] = demos_shown.get(key, 0) + 1
                _print_demo(key, demos_shown[key], gen_roots,
                             prompt_halfbars, chords_per_bar)

            if save_midi_dir is not None and \
                    (save_n_per_key <= 0 or saved_per_key.get(key, 0) < save_n_per_key):
                saved_per_key[key] = saved_per_key.get(key, 0) + 1
                window_idx = start + local_i
                out_path = os.path.join(
                    save_midi_dir,
                    f'{midi_prefix}_{window_idx:05d}_key{ROOT_NAMES[key]}.mid',
                )
                # Slice out this sample from every generated subbeat tensor.
                per_sample = [sampled[t][local_i:local_i + 1, :] for t in range(total_beats)]
                try:
                    decode_output(per_sample, save_path=out_path)
                except Exception as e:
                    print(f'  MIDI-save failed for window {window_idx}: {e}')

    stats: dict[int, dict[str, float]] = {}
    for k, results in per_key.items():
        hb   = np.array([r[0] for r in results])
        bass = np.array([r[1] for r in results])
        cov  = np.array([r[2] for r in results])
        stats[k] = {
            'halfbar_acc': float(hb.mean()),
            'bass_acc':    float(bass.mean()),
            'chord_cov':   float(cov.mean()),
            'n':           len(results),
        }
    return stats


def _print_stats_table(title: str, stats: dict[int, dict[str, float]]) -> None:
    print(f'\n── {title} ──')
    print(f'  {"key":<4}  {"n":>4}  {"halfbar_acc":>12}  {"bass_acc":>10}  {"chord_cov":>10}')
    means = {'hb': [], 'bass': [], 'cov': []}
    for k in sorted(stats):
        s = stats[k]
        print(f'  {ROOT_NAMES[k]:<4}  {s["n"]:>4}  {s["halfbar_acc"]:>12.3f}  '
              f'{s["bass_acc"]:>10.3f}  {s["chord_cov"]:>10.3f}')
        means['hb'].append(s['halfbar_acc'])
        means['bass'].append(s['bass_acc'])
        means['cov'].append(s['chord_cov'])
    if means['hb']:
        print(f'  {"MEAN":<4}  {"":>4}  {np.mean(means["hb"]):>12.3f}  '
              f'{np.mean(means["bass"]):>10.3f}  {np.mean(means["cov"]):>10.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt',    default=None,
                   help='Base CP transformer ckpt (only needed for adapter-only .pt)')
    p.add_argument('--adapter_ckpt', default=None,
                   help='Adapter checkpoint. Omit together with --no_adapter to '
                        'evaluate the plain pretrained base model as a baseline.')
    p.add_argument('--no_adapter',   action='store_true',
                   help='Baseline mode: generate with the ORIGINAL pretrained CP '
                        'transformer only — no adapter, no rule conditioning. '
                        'Requires --base_ckpt; --adapter_ckpt is ignored.')
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
    p.add_argument('--paired_chord_seq', action='store_true',
                   help='Must match the flag used during training. Loads the '
                        '.chord_seq.pt sidecar and hands the chord sequence to '
                        'global_sampling as explicit rule conditioning.')
    p.add_argument('--save_midi_dir', type=str, default=None,
                   help='If set, write generated windows as MIDI here '
                        '(named {seen|unseen}_NNNNN_keyX.mid).')
    p.add_argument('--save_n_per_key', type=int, default=5,
                   help='Cap MIDI files saved per key (accuracies are still '
                        'computed over ALL windows). 0 = save every window.')
    p.add_argument('--n_demos_per_key', type=int, default=2,
                   help='Print a chord-root-per-half-bar demo for the first '
                        'N windows in each key. 0 = no demos.')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gen = args.n_gen_beats or (args.window_len - args.n_prompt_beats)
    max_windows = args.max_windows or None

    print('Loading model ...')
    if args.no_adapter:
        # Baseline: the plain pretrained CP transformer, no adapter layers,
        # no rule conditioning. Wrapped so evaluate_dataset can use the same
        # model.base.* / model.global_sampling interface.
        if not args.base_ckpt or not os.path.exists(args.base_ckpt):
            raise SystemExit('--no_adapter requires a valid --base_ckpt')
        from cp_transformer import RoFormerSymbolicTransformer

        class _BaseOnly:
            def __init__(self, base):
                self.base = base
            def global_sampling(self, x, max_seq_len, temperature,
                                 chord_seq=None, **kw):
                # chord_seq accepted and ignored — the base model is unconditioned.
                return self.base.global_sampling(
                    x, max_seq_len=max_seq_len, temperature=temperature)
            def eval(self):
                self.base.eval()
                return self

        base = RoFormerSymbolicTransformer(size=args.model_size,
                                           max_lr=1e-4, with_velocity=False)
        state = torch.load(args.base_ckpt, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        base.load_state_dict(state)
        base.to(device)
        model = _BaseOnly(base)
        print(f'  BASELINE mode: pretrained base only ({args.base_ckpt})')
    else:
        if not args.adapter_ckpt:
            raise SystemExit('--adapter_ckpt is required (or pass --no_adapter '
                             'for the pretrained-base baseline)')
        model = load_model(args.base_ckpt, args.adapter_ckpt,
                           args.model_size, args.adapter_rank, args.n_skip,
                           args.bidirectional, args.encoder_injected,
                           args.encoder_type, args.rule_mode, args.approach,
                           chords_per_bar=args.chords_per_bar,
                           chord_seq_conditioning=args.paired_chord_seq,
                           device=device)
    model.eval()

    for label, path in (('seen', args.seen_data),
                         ('unseen', args.unseen_data)):
        if path is None:
            continue
        print(f'\nLoading {label.upper()} keys: {path}')
        windows, keys, chord_seq = _load_windows(path, args.window_len)
        print(f'  {len(windows)} windows')
        if args.paired_chord_seq and chord_seq is None:
            raise SystemExit(f'--paired_chord_seq set but no .chord_seq.pt sidecar '
                             f'found next to {path}')
        # Only pass chord_seq to the model when we're in chord-seq mode; otherwise
        # global_sampling handles rule conditioning itself from the prompt/key.
        cs_for_model = chord_seq if args.paired_chord_seq else None

        midi_dir = None
        if args.save_midi_dir is not None:
            midi_dir = os.path.join(args.save_midi_dir, label)

        stats = evaluate_dataset(
            model, windows, keys,
            n_prompt_beats  =args.n_prompt_beats,
            n_gen_beats     =n_gen,
            temperature     =args.temperature,
            chords_per_bar  =args.chords_per_bar,
            device          =device,
            batch_size      =args.batch_size,
            max_windows     =max_windows,
            chord_seq       =cs_for_model,
            save_midi_dir   =midi_dir,
            midi_prefix     =label,
            n_demos_per_key =args.n_demos_per_key,
            save_n_per_key  =args.save_n_per_key,
        )
        _print_stats_table(f'{label.upper()} keys  (prompt={args.n_prompt_beats}, '
                            f'gen={n_gen}, T={args.temperature})', stats)


if __name__ == '__main__':
    main()
