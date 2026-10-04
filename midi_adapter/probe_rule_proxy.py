#!/usr/bin/env python3
"""Is the AR->rule encoder actually working?

Generation quality confounds three things: whether ar_to_rule can name a chord
from the base's hidden state, whether the compiled program then recovers the
key, and whether the adapter renders it. This probe measures the first two
directly, teacher-forced, with no sampling involved.

Reported per dataset:
  proxy root acc        argmax(ar_to_rule(h)) == the true chord root at t
  proxy root @ phase-0  the same, restricted to the positions the head reads
  retrieved key acc     the program's key region == the window's true key
  out acc               the program's out region == the true root at t
  chance                1/12 = 0.083

If "proxy root acc" sits at chance, the encoder is the problem and no amount of
adapter training will help. If the proxy is good but "retrieved key acc" is
not, the retrieval step is the problem -- which is what --rule_program mlp_only
exists to test.

Usage:
  python -m midi_adapter.probe_rule_proxy \\
      --base_ckpt checkpoints/cp_transformer_v0.42_...ckpt \\
      --adapter_ckpt checkpoints/<run>/<run>.by_val_loss....ckpt \\
      --data /l/users/xinyue.li/data/pop909_ivvi_w1/direct_val_seenkeys.pt \\
      --bidirectional --rule_attention --rule_program full \\
      --rule_from_layer 0 --ar_to_rule_hidden 256 --proxy_activation sigmoid \\
      --positional_qk
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.evaluate_cp_yinyang import load_model
from midi_adapter.evaluate_on_real import _load_windows
from rasp_program.sequence_rule import OFFSETS

N_ROOTS, N_POS = 12, 4


@torch.no_grad()
def probe(model, windows, keys, chords_per_bar, batch_size, device):
    spc = 16 // chords_per_bar
    rm  = model.rule_model
    has_out = hasattr(rm, 'O0')
    acc = {k: [0, 0] for k in
           ('proxy', 'proxy_p0', 'key', 'out')}          # [hits, n]

    for s in range(0, len(windows), batch_size):
        x = windows[s:s + batch_size].to(device).long()
        k = torch.tensor(keys[s:s + batch_size], device=device)
        valid = k >= 0
        if not valid.any():
            continue
        x, k = x[valid], k[valid]
        x_proc = model.base.preprocess(x, torch.zeros(len(x), dtype=torch.long,
                                                      device=device))
        B, T = x_proc.shape[0], x_proc.shape[1]

        # the stream entering the stack — exactly what forward() builds
        h, _ = model.base.local_encode(x_proc)
        h = h.view(B, T, model.base.hidden_size)
        sos = model.base.global_sos.view(1, 1, -1).expand(B, 1, -1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

        if model.rule_from_layer > 0:                    # read after layer k
            mask = model.base.buffered_future_mask(h)
            for i, layer in enumerate(model.base.model.layer):
                h = layer(h, attention_mask=mask)[0]
                if i + 1 == model.rule_from_layer:
                    break

        idx = 0 if model.rule_from_layer >= 0 else 0     # probe the first proj
        est = model._activate_root(model.ar_to_rule[idx](h))[..., :N_ROOTS]
        r   = model._rule_proxy(h, idx)

        phase = (torch.arange(T, device=device) // spc) % N_POS
        off   = torch.tensor(OFFSETS, device=device)[phase]
        true_root = (k[:, None] + off[None, :]) % N_ROOTS            # (B, T)

        p = est.argmax(-1)
        acc['proxy'][0] += int((p == true_root).sum()); acc['proxy'][1] += p.numel()
        m = (phase == 0)[None, :].expand(B, -1)
        acc['proxy_p0'][0] += int((p[m] == true_root[m]).sum())
        acc['proxy_p0'][1] += int(m.sum())

        if has_out:
            kk = r[..., rm.K0:rm.K0 + N_ROOTS].argmax(-1)
            oo = r[..., rm.O0:rm.O0 + N_ROOTS].argmax(-1)
            acc['key'][0] += int((kk == k[:, None]).sum()); acc['key'][1] += kk.numel()
            acc['out'][0] += int((oo == true_root).sum());  acc['out'][1] += oo.numel()
        else:                                            # retrieve program
            kk = r[..., N_ROOTS:2 * N_ROOTS].argmax(-1)
            acc['key'][0] += int((kk == k[:, None]).sum()); acc['key'][1] += kk.numel()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt', required=True)
    p.add_argument('--adapter_ckpt', required=True)
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--window_len', type=int, default=64)
    p.add_argument('--chords_per_bar', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_windows', type=int, default=0)
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip', type=int, default=1)
    p.add_argument('--approach', type=str, default='chord')
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--rule_attention', action='store_true')
    p.add_argument('--rule_program', type=str, default='retrieve')
    p.add_argument('--rule_input', type=str, default='root')
    p.add_argument('--rule_heads', type=int, default=1)
    p.add_argument('--rule_from_layer', type=int, default=-1)
    p.add_argument('--ar_to_rule_hidden', type=int, default=0)
    p.add_argument('--proxy_activation', type=str, default='none')
    p.add_argument('--proxy_temp', type=float, default=1.0)
    p.add_argument('--positional_qk', action='store_true')
    p.add_argument('--qk_content_residual', action='store_true')
    p.add_argument('--content_residual', type=str, default='none')
    p.add_argument('--lora_rank', type=int, default=0)
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(a.base_ckpt, a.adapter_ckpt, a.model_size, a.adapter_rank,
                       a.n_skip, a.bidirectional, False, 'embedding', 'current',
                       a.approach, chords_per_bar=a.chords_per_bar,
                       chord_seq_conditioning=False, lora_rank=a.lora_rank,
                       positional_qk=a.positional_qk,
                       qk_content_residual=a.qk_content_residual,
                       content_residual=a.content_residual,
                       rule_attention=a.rule_attention,
                       rule_program=a.rule_program, rule_input=a.rule_input,
                       rule_heads=a.rule_heads,
                       rule_from_layer=a.rule_from_layer,
                       ar_to_rule_hidden=a.ar_to_rule_hidden,
                       proxy_activation=a.proxy_activation,
                       proxy_temp=a.proxy_temp, device=dev)
    model.eval()
    print(f'rule model : {type(model.rule_model).__name__}  '
          f'ar_to_rule : {len(model.ar_to_rule)} x '
          f'{type(model.ar_to_rule[0]).__name__}  '
          f'read at layer {model.rule_from_layer}')

    print(f'\n  {"dataset":<34}{"proxy":>9}{"proxy@p0":>11}'
          f'{"key":>9}{"out":>9}{"n win":>8}')
    print('  ' + '-' * 80)
    for path in a.data:
        w, k, _ = _load_windows(path, a.window_len)
        if a.max_windows:
            w, k = w[:a.max_windows], k[:a.max_windows]
        acc = probe(model, w, k, a.chords_per_bar, a.batch_size, dev)
        f = lambda n: (f'{acc[n][0]/acc[n][1]:.3f}' if acc[n][1] else '—')
        print(f'  {os.path.basename(path):<34}{f("proxy"):>9}{f("proxy_p0"):>11}'
              f'{f("key"):>9}{f("out"):>9}{len(w):>8}')
    print(f'  {"chance (1/12)":<34}{0.0833:>9.3f}{0.0833:>11.3f}'
          f'{0.0833:>9.3f}{0.0833:>9.3f}')
    print('\n  proxy at chance  -> the encoder cannot name chords; the adapter '
          'cannot fix that')
    print('  proxy good, key poor -> the retrieval step is the problem '
          '(try --rule_program mlp_only)')


if __name__ == '__main__':
    main()
