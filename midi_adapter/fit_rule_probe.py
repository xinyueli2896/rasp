#!/usr/bin/env python3
"""How much chord-root information is in the frozen base's hidden states?

The jointly-trained ar_to_rule settled at ~0.34 per-position root accuracy.
That has two very different explanations:

  (a) the information is not there at that depth  -> read deeper
  (b) joint training failed to extract it         -> fix the optimisation

This separates them. The base is frozen, so its hidden states are fixed data.
Fit a standalone probe on (h at layer L) -> chord root, supervised, with no LM
loss and no adapter in the way. That probe's accuracy is the CEILING available
at that layer. Compare it to what joint training achieved:

  probe >> joint  ->  the information was there; joint training is the problem
  probe ~  joint  ->  joint training is already extracting what exists

Sweeping L also answers "which layer should --rule_from_layer read?" without
running a single adapter training.

Usage:
  python -m midi_adapter.fit_rule_probe \\
      --base_ckpt checkpoints/cp_transformer_v0.42_...ckpt \\
      --data /l/users/xinyue.li/data/pop909_ivvi_w1/train_all_keys_seenkeys.pt \\
      --val  /l/users/xinyue.li/data/pop909_ivvi_w1/direct_val_seenkeys.pt \\
      --layers 0 1 2 6 12
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.evaluate_on_real import _load_windows
from rasp_program.sequence_rule import OFFSETS

N_ROOTS, N_POS = 12, 4


@torch.no_grad()
def hidden_at(base, windows, keys, layers, chords_per_bar, batch_size, device,
              max_windows=0):
    """Collect h at each requested layer, plus the true root, for every position.

    layer 0 = the stream ENTERING the stack (after the sos shift), which is
    exactly what --rule_from_layer 0 reads.
    """
    spc = 16 // chords_per_bar
    out = {L: [] for L in layers}
    tgt = []
    if max_windows:
        windows, keys = windows[:max_windows], keys[:max_windows]

    for s in range(0, len(windows), batch_size):
        x = windows[s:s + batch_size].to(device).long()
        k = torch.tensor(keys[s:s + batch_size], device=device)
        ok = k >= 0
        if not ok.any():
            continue
        x, k = x[ok], k[ok]
        xp = base.preprocess(x, torch.zeros(len(x), dtype=torch.long, device=device))
        B, T = xp.shape[0], xp.shape[1]

        h, _ = base.local_encode(xp)
        h = h.view(B, T, base.hidden_size)
        sos = base.global_sos.view(1, 1, -1).expand(B, 1, -1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        if 0 in out:
            out[0].append(h.reshape(-1, base.hidden_size).cpu())

        mask = base.buffered_future_mask(h)
        for i, layer in enumerate(base.model.layer):
            h = layer(h, attention_mask=mask)[0]
            if (i + 1) in out:
                out[i + 1].append(h.reshape(-1, base.hidden_size).cpu())

        phase = (torch.arange(T, device=device) // spc) % N_POS
        off   = torch.tensor(OFFSETS, device=device)[phase]
        root  = (k[:, None] + off[None, :]) % N_ROOTS
        tgt.append(root.reshape(-1).cpu())

    return ({L: torch.cat(v) for L, v in out.items()}, torch.cat(tgt))


def fit(X, y, Xv, yv, hidden, epochs, lr, device, seed=0):
    torch.manual_seed(seed)
    d = X.shape[1]
    net = (nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, N_ROOTS))
           if hidden else nn.Linear(d, N_ROOTS)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    X, y, Xv, yv = X.to(device), y.to(device), Xv.to(device), yv.to(device)
    n, bs = len(X), 4096
    best = (0.0, 9.9)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lo = net(Xv)
            acc = float((lo.argmax(-1) == yv).float().mean())
            ce  = float(F.cross_entropy(lo, yv))
        if acc > best[0]:
            best = (acc, ce)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt', required=True)
    p.add_argument('--data', required=True, help='windows to FIT the probe on')
    p.add_argument('--val',  required=True, help='held-out windows to score on')
    p.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2, 6, 12])
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--window_len', type=int, default=64)
    p.add_argument('--chords_per_bar', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_windows', type=int, default=1500)
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--joint_acc', type=float, default=0.34,
                   help='what the jointly-trained encoder achieved, for the '
                        'side-by-side (default 0.34 = the L0 run).')
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = RoFormerSymbolicTransformer(size=a.model_size, max_lr=1e-4,
                                       with_velocity=False)
    st = torch.load(a.base_ckpt, map_location='cpu')
    st = st.get('state_dict', st)
    if any(k.startswith('model.base.') for k in st):
        st = {k[len('model.base.'):]: v for k, v in st.items()
              if k.startswith('model.base.')}
    base.load_state_dict(st); base.to(dev).eval()

    print(f'collecting hidden states at layers {a.layers} ...')
    wtr, ktr, _ = _load_windows(a.data, a.window_len)
    wva, kva, _ = _load_windows(a.val,  a.window_len)
    Htr, ytr = hidden_at(base, wtr, ktr, a.layers, a.chords_per_bar,
                         a.batch_size, dev, a.max_windows)
    Hva, yva = hidden_at(base, wva, kva, a.layers, a.chords_per_bar,
                         a.batch_size, dev)
    print(f'  fit on {len(ytr):,} positions, score on {len(yva):,}')

    print(f'\n  {"layer":>6}{"linear acc":>13}{"MLP acc":>10}{"MLP CE":>9}'
          f'{"vs joint":>11}')
    print('  ' + '-' * 50)
    for L in a.layers:
        la, _  = fit(Htr[L], ytr, Hva[L], yva, 0,        a.epochs, a.lr, dev)
        ma, mc = fit(Htr[L], ytr, Hva[L], yva, a.hidden, a.epochs, a.lr, dev)
        tag = 'layer 0 = pre-stack' if L == 0 else ''
        print(f'  {L:>6}{la:>13.3f}{ma:>10.3f}{mc:>9.3f}'
              f'{ma - a.joint_acc:>+11.3f}   {tag}')
    print(f'  {"chance":>6}{1/N_ROOTS:>13.3f}{1/N_ROOTS:>10.3f}'
          f'{float(np.log(N_ROOTS)):>9.3f}')
    print(f'\n  joint training reached {a.joint_acc:.3f} at layer 0.')
    print('  probe >> joint at layer 0  -> the information was there and joint')
    print('                                training failed to extract it')
    print('  probe ~  joint at layer 0  -> joint training is already at the')
    print('                                ceiling; read a deeper layer instead')
    print('  the layer column also says where --rule_from_layer should read,')
    print('  without running an adapter training to find out.')


if __name__ == '__main__':
    main()
