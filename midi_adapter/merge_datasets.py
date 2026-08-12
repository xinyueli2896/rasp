"""
Concatenate two or more CP-tensor datasets (same format) into one.

All inputs must share the same window length and subseq width (i.e. the same
--max_polyphony). Sidecars are merged consistently:
    .pt                   concatenated along dim 0
    .length.pt            concatenated
    .pitch_shift_range.pt concatenated
    .beat_chords.pt       list concatenation
    .keys.pt              concatenated (must exist in ALL inputs or none)
    .chord_seq.pt         concatenated (must exist in ALL inputs or none;
                          chord-position count must match)
    .txt                  renumbered line concatenation, source tagged

Usage
-----
    python -m midi_adapter.merge_datasets \\
        --inputs  /data/pop909_direct_train.pt /data/nottingham_direct_train.pt \\
        --out_pt  /data/combined_direct_train
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sidecar(path: str, suffix: str) -> str:
    return path[:-3] + suffix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True,
                   help='Two or more dataset .pt files to concatenate (in order).')
    p.add_argument('--out_pt', required=True,
                   help='Output prefix (no .pt extension needed; .pt is appended).')
    args = p.parse_args()

    out_prefix = args.out_pt[:-3] if args.out_pt.endswith('.pt') else args.out_pt

    datas, lengths, psrs, chords, keys, chord_seqs, txts = [], [], [], [], [], [], []
    have_keys      = None
    have_chord_seq = None
    subseq_width   = None
    window_len     = None

    for path in args.inputs:
        if not os.path.exists(path):
            raise SystemExit(f'missing input: {path}')
        d = torch.load(path, weights_only=True)
        l = torch.load(_sidecar(path, '.length.pt'), weights_only=True)
        r = torch.load(_sidecar(path, '.pitch_shift_range.pt'), weights_only=True).reshape(-1, 2)
        c = torch.load(_sidecar(path, '.beat_chords.pt'), weights_only=True)

        if subseq_width is None:
            subseq_width = d.shape[-1]
        elif d.shape[-1] != subseq_width:
            raise SystemExit(f'{path}: subseq width {d.shape[-1]} != {subseq_width} '
                             f'(different --max_polyphony?)')
        wl = int(l[0])
        if window_len is None:
            window_len = wl
        elif wl != window_len:
            raise SystemExit(f'{path}: window length {wl} != {window_len}')
        if not bool((l == wl).all()):
            raise SystemExit(f'{path}: variable window lengths not supported')

        k_path, cs_path = _sidecar(path, '.keys.pt'), _sidecar(path, '.chord_seq.pt')
        k  = torch.load(k_path,  weights_only=True).long() if os.path.exists(k_path)  else None
        cs = torch.load(cs_path, weights_only=True).long() if os.path.exists(cs_path) else None
        if have_keys is None:
            have_keys = k is not None
        elif have_keys != (k is not None):
            raise SystemExit('some inputs have .keys.pt and some do not — cannot merge')
        if have_chord_seq is None:
            have_chord_seq = cs is not None
        elif have_chord_seq != (cs is not None):
            raise SystemExit('some inputs have .chord_seq.pt and some do not — cannot merge')
        if cs is not None and chord_seqs and cs.shape[1] != chord_seqs[0].shape[1]:
            raise SystemExit(f'{path}: chord_seq has {cs.shape[1]} positions, '
                             f'expected {chord_seqs[0].shape[1]}')

        tag = os.path.splitext(os.path.basename(path))[0]
        txt_path = _sidecar(path, '.txt')
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                src_lines = [ln.rstrip('\n').split('\t', 1) for ln in f if ln.strip()]
            txts.append([(tag, rel) for _, rel in src_lines])
        else:
            txts.append([(tag, f'window_{i}') for i in range(len(l))])

        datas.append(d); lengths.append(l); psrs.append(r); chords.extend(c)
        if k  is not None: keys.append(k)
        if cs is not None: chord_seqs.append(cs)
        print(f'  {path}: {len(l)} windows')

    n_total = sum(len(l) for l in lengths)
    torch.save(torch.cat(datas,   dim=0), f'{out_prefix}.pt')
    torch.save(torch.cat(lengths, dim=0), f'{out_prefix}.length.pt')
    torch.save(torch.cat(psrs,    dim=0), f'{out_prefix}.pitch_shift_range.pt')
    torch.save(chords,                    f'{out_prefix}.beat_chords.pt')
    if have_keys:
        torch.save(torch.cat(keys, dim=0),       f'{out_prefix}.keys.pt')
    if have_chord_seq:
        torch.save(torch.cat(chord_seqs, dim=0), f'{out_prefix}.chord_seq.pt')

    with open(f'{out_prefix}.txt', 'w') as f:
        i = 0
        for group in txts:
            for tag, rel in group:
                f.write(f'{i}\t{tag}:{rel}\n')
                i += 1

    print(f'\nMerged {len(args.inputs)} datasets → {out_prefix}.pt  ({n_total} windows)')


if __name__ == '__main__':
    main()
