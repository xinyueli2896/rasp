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
    # The 12 dims carry no intrinsic labelling. The program applies a CYCLIC
    # SHIFT, so an encoding of root + c for any fixed c is equally valid: the
    # shift still composes correctly and v_proj decodes by subtracting c.
    # Scoring only c = 0 would call a perfectly good encoder chance-level, so
    # tally every rotation and report the best.
    rot = {k: np.zeros(N_ROOTS, dtype=np.int64) for k in ('proxy', 'key')}
    # Rotation-invariant: does the proxy get the INTERVALS right, whatever its
    # absolute labelling? This is what "has it learned the rule" really asks.
    interval = [0, 0]
    # Hypothesis for a BELOW-chance interval: at rule_from_layer 0 no
    # self-attention has run, so h[t] encodes little beyond subbeat t-1 and the
    # encoder tracks the NOTE sounding rather than the chord. Test it: how often
    # is argmax(proxy[t]) a pitch class actually present at t-1? Compare against
    # the density of that chromagram, which is what a random argmax would score.
    local = [0, 0]
    dens  = [0.0, 0]
    # proxy ~ 0.34 is close to 1/3, which is what "name ANY sounding pitch
    # class at random" scores when a triad is sounding. If the encoder merely
    # learned to echo a note that is present, it gets the root by luck at
    # 1/|pcs|. Compute that baseline exactly, per position, so the comparison
    # is not eyeballed: sound_base = mean over t of 1/|pcs(t-1)| when the true
    # root is among them, else 0.
    sound_base = [0.0, 0]
    root_heard = [0, 0]
    # interval near 1/T is the fingerprint of a COLLAPSED encoder: position 0
    # reads the sos vector and every other position reads content, so if the
    # proxy emits one constant for t>0 only t=0 matches trivially. Measure it:
    # how much of each window sits on its own modal argmax, and how many
    # distinct values the argmax takes at all.
    collapse = [0, 0]
    distinct = [0, 0]
    # distinct ~ 4 with on-mode ~ 0.5 points at a PHASE LOOKUP: the encoder
    # emitting one value per bar phase and ignoring the music entirely. That is
    # the degenerate optimum -- it hands the LM a free 4-phase clock, which
    # genuinely helps predict notes, without ever reading a chord. Test it:
    # how much of each phase class sits on that class's own modal argmax?
    byphase = [0, 0]
    ph_dist = [0, 0]

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
        for c in range(N_ROOTS):
            rot['proxy'][c] += int((((p - c) % N_ROOTS) == true_root).sum())
        # intervals: compare each position to position 0 of its own window
        chroma = model._extract_chromagram(x_proc)          # (B, T, 12)
        prev   = torch.roll(chroma, 1, dims=1); prev[:, 0] = 0
        hit    = prev.gather(-1, p.unsqueeze(-1)).squeeze(-1) > 0
        local[0] += int(hit[:, 1:].sum()); local[1] += hit[:, 1:].numel()
        dens[0]  += float(prev[:, 1:].sum() / N_ROOTS); dens[1] += prev[:, 1:].shape[0] * prev[:, 1:].shape[1]

        for b in range(p.shape[0]):
            vals, cnt = torch.unique(p[b, 1:], return_counts=True)
            collapse[0] += int(cnt.max()); collapse[1] += int(p.shape[1] - 1)
            distinct[0] += int(len(vals));  distinct[1] += 1

        for b in range(p.shape[0]):
            for ph in range(N_POS):
                sel = p[b][phase == ph]
                if sel.numel() == 0:
                    continue
                v, c = torch.unique(sel, return_counts=True)
                byphase[0] += int(c.max()); byphase[1] += int(sel.numel())
                ph_dist[0] += int(len(v));  ph_dist[1] += 1

        npc  = prev.sum(-1).clamp(min=1)                       # |pcs| at t-1
        rt   = prev.gather(-1, true_root.unsqueeze(-1)).squeeze(-1) > 0
        base = torch.where(rt, 1.0 / npc, torch.zeros_like(npc))
        sound_base[0] += float(base[:, 1:].sum()); sound_base[1] += base[:, 1:].numel()
        root_heard[0] += int(rt[:, 1:].sum());     root_heard[1] += rt[:, 1:].numel()

        d_pred = (p - p[:, :1]) % N_ROOTS
        d_true = (true_root - true_root[:, :1]) % N_ROOTS
        interval[0] += int((d_pred == d_true).sum()); interval[1] += d_pred.numel()
        m = (phase == 0)[None, :].expand(B, -1)
        acc['proxy_p0'][0] += int((p[m] == true_root[m]).sum())
        acc['proxy_p0'][1] += int(m.sum())

        if has_out:
            kk = r[..., rm.K0:rm.K0 + N_ROOTS].argmax(-1)
            oo = r[..., rm.O0:rm.O0 + N_ROOTS].argmax(-1)
            acc['key'][0] += int((kk == k[:, None]).sum()); acc['key'][1] += kk.numel()
            acc['out'][0] += int((oo == true_root).sum());  acc['out'][1] += oo.numel()
            for c in range(N_ROOTS):
                rot['key'][c] += int((((kk - c) % N_ROOTS) == k[:, None]).sum())
        else:                                            # retrieve program
            kk = r[..., N_ROOTS:2 * N_ROOTS].argmax(-1)
            acc['key'][0] += int((kk == k[:, None]).sum()); acc['key'][1] += kk.numel()
            for c in range(N_ROOTS):
                rot['key'][c] += int((((kk - c) % N_ROOTS) == k[:, None]).sum())
    return (acc, rot, interval, local, dens, collapse, distinct, byphase,
            ph_dist, sound_base, root_heard)


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

    print(f'\n  {"dataset":<34}{"proxy":>9}{"proxy@p0":>11}{"key":>9}{"out":>9}'
          f'{"best-rot":>10}{"c":>5}{"key-rot":>9}{"c":>5}{"interval":>11}'
          f'{"local-pc":>9}{"density":>9}{"on-mode":>10}{"distinct":>10}'
          f'{"by-phase":>10}{"ph-dist":>9}{"echo-base":>11}{"heard":>9}{"n":>7}')
    print('  ' + '-' * 168)
    for path in a.data:
        w, k, _ = _load_windows(path, a.window_len)
        if a.max_windows:
            w, k = w[:a.max_windows], k[:a.max_windows]
        (acc, rot, interval, local, dens, collapse, distinct, byphase, ph_dist,
         sbase, rheard) = probe(model, w, k, a.chords_per_bar, a.batch_size, dev)
        f = lambda n: (f'{acc[n][0]/acc[n][1]:.3f}' if acc[n][1] else '—')
        bp = int(rot['proxy'].argmax()); bk = int(rot['key'].argmax())
        rp = rot['proxy'][bp] / max(acc['proxy'][1], 1)
        rk = rot['key'][bk]   / max(acc['key'][1], 1)
        iv = interval[0] / max(interval[1], 1)
        print(f'  {os.path.basename(path):<34}{f("proxy"):>9}{f("proxy_p0"):>11}'
              f'{f("key"):>9}{f("out"):>9}'
              f'{rp:>10.3f}{bp:>5}{rk:>9.3f}{bk:>5}{iv:>11.3f}'
              f'{local[0]/max(local[1],1):>9.3f}{dens[0]/max(dens[1],1):>9.3f}'
              f'{collapse[0]/max(collapse[1],1):>10.3f}'
              f'{distinct[0]/max(distinct[1],1):>10.1f}'
              f'{byphase[0]/max(byphase[1],1):>10.3f}'
              f'{ph_dist[0]/max(ph_dist[1],1):>9.1f}'
              f'{sbase[0]/max(sbase[1],1):>11.3f}'
              f'{rheard[0]/max(rheard[1],1):>9.3f}{len(w):>7}')
    # Baselines differ per column and getting this wrong is easy. A CONSTANT
    # predictor already scores every position where OFFSETS[phase] == OFFSETS[0],
    # which is 2 of 4 phases, so the interval column must beat 0.5, not 1/12.
    const_iv = sum(1 for o in OFFSETS if o == OFFSETS[0]) / len(OFFSETS)
    print(f'  {"chance (1/12)":<34}{0.0833:>9.3f}{0.0833:>11.3f}'
          f'{0.0833:>9.3f}{0.0833:>9.3f}{0.0833:>10.3f}{"":>5}'
          f'{0.0833:>9.3f}{"":>5}{"":>11}')
    print(f'  {"constant predictor":<34}{"":>9}{"":>11}{"":>9}{"":>9}'
          f'{"":>10}{"":>5}{"":>9}{"":>5}{const_iv:>11.3f}')
    print()
    print('  best-rot  = accuracy under the most favourable global rotation c.')
    print('              The program applies a cyclic shift, so root + c is an')
    print('              equally valid encoding and c is unidentifiable from the')
    print('              LM loss alone. If best-rot >> proxy, the encoder works')
    print('              in a rotated frame and only the LABELLING is arbitrary.')
    print('  interval  = does the proxy get root[t] - root[0] right? Fully')
    print('              rotation-invariant: this is "has it learned the rule".')
    print(f'              BEAT {const_iv:.3f}, not 1/12 — a constant predictor')
    print('              already gets every phase whose offset equals phase 0.')
    print()
    print('  all three at chance   -> the encoder learned nothing')
    print('  best-rot high, proxy low -> it works, just in a rotated frame')
    print('  interval high, best-rot low -> relative structure without a stable')
    print('                                 frame, i.e. no consistent key')
    print()
    print('  local-pc  = is argmax(proxy[t]) a pitch class actually sounding at')
    print('              t-1?  density = what a RANDOM argmax would score, i.e.')
    print('              the mean fraction of the 12 classes present. local-pc')
    print('              well above density means the encoder is tracking the')
    print('              NOTE under the cursor rather than the chord -- which is')
    print('              all that h[t] contains when read before the stack.')
    print('  on-mode   = share of each window sitting on its own modal argmax;')
    print('              distinct = how many argmax values a window uses at all.')
    print('              on-mode ~1.0 with distinct ~1 is a COLLAPSED encoder:')
    print('              it emits one vector regardless of the music, and the')
    print('              program turns that into a fixed per-phase bias -- useful')
    print('              for the LM loss, and reachable without reading anything.')
    print('  by-phase  = share of each PHASE CLASS on that class\'s modal argmax;')
    print('              ph-dist = values used within a phase class. by-phase ~1.0')
    print('              with ph-dist ~1 means the encoder is a PHASE LOOKUP: one')
    print('              value per bar position, music ignored. That is the same')
    print('              degenerate clock as collapse, one step less obvious.')
    print('  echo-base = what "name any SOUNDING pitch class at random" scores,')
    print('              computed exactly per position as 1/|pcs| when the true')
    print('              root is audible, else 0. heard = how often the root is')
    print('              audible at all. If proxy ~ echo-base the encoder only')
    print('              learned to ECHO a note that is present and gets the root')
    print('              by luck; proxy >> echo-base means it prefers the root.')


if __name__ == '__main__':
    main()
