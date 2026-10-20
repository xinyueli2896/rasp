"""Is the adapter's cross-attention branch alive, per run and per layer?

With gate_init=0.0 the branch starts at exactly zero contribution, so the
only thing that can open it is the gradient reaching that one scalar. On a
stem with no other path to reduce loss -- frozen host, lora_rank=0 -- a gate
that fails to open early can stay shut for the whole run, and the result is
the bare pretrained host wearing an inert adapter.

The gate alone does not settle that. In CPYinyangCrossAttention the branch
contributes out_proj(out) * gate, so a small gate with a large out_proj is
still a live branch. What matters is the product, reported here as `scale`:

    scale_i = |gate_i| * ||W_out_i||_fro / sqrt(d_model)

The sqrt(d_model) divisor makes `scale` comparable across model sizes; it is
not an exact bound on the branch's output norm, just a consistent proxy.

Two checkpoints are read per run, because they answer different questions:
  selected  the one the table used (highest unseen_acc) -- was the branch
            open at the point we scored?
  last      end of training -- did the branch ever open at all?

Usage
-----
  python -m midi_adapter.gate_report checkpoints_bar
  python -m midi_adapter.gate_report checkpoints_bar checkpoints_bar_h1 -v
  python -m midi_adapter.gate_report checkpoints_bar --runs 'cp_yy_frozen*'
"""
from __future__ import annotations

import argparse
import fnmatch
import math
import os
import re
import sys

import torch


# Not [0-9.]+ -- that is greedy enough to swallow the '.' before 'ckpt' and
# then fails to parse ('0.7786.').
ACC_RE = re.compile(r'unseen_acc=(\d+(?:\.\d+)?)')
LAYER_RE = re.compile(r'\.yinyang_attn\.(\d+)\.')

# Below this the branch contributes essentially nothing and the run is the
# bare host. It is a reporting threshold, not a result -- a run just above it
# is not thereby working.
DEAD = 1e-3


def _pick(ckpt_dir: str):
    """(selected, last) checkpoint paths; either may be None."""
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return None, None
    last = os.path.join(ckpt_dir, 'last.ckpt') if 'last.ckpt' in names else None
    scored = []
    for n in names:
        if not n.endswith('.ckpt') or n == 'last.ckpt':
            continue
        m = ACC_RE.search(n)
        if m:
            scored.append((float(m.group(1)), n))
    # No unseen_acc in the filenames (e.g. the finetune baseline, which is
    # selected on val_loss and has no adapter anyway) -> fall back to the
    # newest file so the run is still reported rather than skipped.
    if scored:
        sel = os.path.join(ckpt_dir, max(scored)[1])
    else:
        cands = [n for n in names if n.endswith('.ckpt') and n != 'last.ckpt']
        sel = (os.path.join(ckpt_dir, max(
            cands, key=lambda n: os.path.getmtime(os.path.join(ckpt_dir, n))))
            if cands else None)
    return sel, last


def _gates(path: str):
    """[(layer, gate, scale)] for one checkpoint, ordered by layer."""
    blob = torch.load(path, map_location='cpu', weights_only=False)
    sd = blob.get('state_dict', blob)

    gates, wnorm, dmodel = {}, {}, {}
    for k, v in sd.items():
        m = LAYER_RE.search(k)
        if not m:
            continue
        i = int(m.group(1))
        if k.endswith('.gate'):
            gates[i] = float(v.reshape(-1)[0])
        elif k.endswith('.out_proj.weight'):
            wnorm[i] = float(v.float().norm())
            dmodel[i] = int(v.shape[0])

    rows = []
    for i in sorted(gates):
        g = gates[i]
        w = wnorm.get(i)
        d = dmodel.get(i)
        scale = abs(g) * w / math.sqrt(d) if (w and d) else float('nan')
        rows.append((i, g, scale))
    return rows


def _summarise(rows):
    s = [r[2] for r in rows if not math.isnan(r[2])]
    g = [abs(r[1]) for r in rows]
    if not g:
        return None
    return {
        'n': len(rows),
        'gate_max': max(g), 'gate_mean': sum(g) / len(g), 'gate_min': min(g),
        'scale_max': max(s) if s else float('nan'),
        'scale_mean': sum(s) / len(s) if s else float('nan'),
        'dead': sum(1 for v in (s or g) if v < DEAD),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ckpt_dirs', nargs='+',
                   help='Directories holding one subdirectory per run.')
    p.add_argument('--runs', default='*',
                   help='Glob over run names (default all).')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Per-layer gate and scale for every run.')
    args = p.parse_args()

    print('scale = |gate| * ||W_out||_F / sqrt(d_model); '
          f'a branch under {DEAD:g} contributes nothing')
    print()
    hdr = (f'{"run":<34} {"ckpt":<9} {"n":>3} {"gate max":>9} {"gate mean":>10} '
           f'{"scale max":>10} {"scale mean":>11} {"dead":>5}')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))

    found = 0
    for root in args.ckpt_dirs:
        if not os.path.isdir(root):
            print(f'  (no such directory: {root})', file=sys.stderr)
            continue
        for run in sorted(os.listdir(root)):
            d = os.path.join(root, run)
            if not os.path.isdir(d) or not fnmatch.fnmatch(run, args.runs):
                continue
            sel, last = _pick(d)
            printed_any = False
            for tag, path in (('selected', sel), ('last', last)):
                if path is None:
                    continue
                try:
                    rows = _gates(path)
                except Exception as e:                     # noqa: BLE001
                    print(f'{run:<34} {tag:<9} load failed: {type(e).__name__}: {e}')
                    continue
                st = _summarise(rows)
                if st is None:
                    continue          # no adapter in this checkpoint (baseline)
                found += 1
                printed_any = True
                print(f'{run:<34} {tag:<9} {st["n"]:>3} '
                      f'{st["gate_max"]:>9.5f} {st["gate_mean"]:>10.5f} '
                      f'{st["scale_max"]:>10.5f} {st["scale_mean"]:>11.5f} '
                      f'{st["dead"]:>5}')
                if args.verbose:
                    # .3g, not .4f: a dead gate is ~1e-5 and fixed-point
                    # rounds it to 0.0000, hiding the thing being measured.
                    print(f'{"":<34} {"":<9} ' + '  '.join(
                        f'L{i}:{g:+.3g}/{s:.3g}' for i, g, s in rows))
            if printed_any:
                print('-' * len(hdr))

    print('=' * len(hdr))
    if not found:
        print('No adapter gates found. Checked for keys matching '
              "'*.yinyang_attn.<i>.gate' -- are these adapter runs?")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
