"""
X11 round 1, step 1: hidden states for the two cadence probes.

Round 1 asks whether a model WITHOUT an adapter already carries the cadence in
its residual stream. Two targets, both read at the step just before a bar's
downbeat -- the position whose output predicts the downbeat, so it has heard
every earlier bar and nothing of the bar being asked about:

    roman       the roman numeral bar b will play     3 classes: I, IV, V
                (minor tonics: i, iv, V share the classes of I, IV, V)
    next_chord  the chord bar b will play             24 triads

Both labels are exact on the v6 spans, which are selected so every bar plays
CADENCE[tonic][bar % 4].

What is saved, per model, for every (window, bar b = 1..W-1):

    <out>/<model>/feats.npy   float16 (N, n_layers + 1, hidden)
        layer 0  = the sequence fed to the global encoder: the local-encoder
                   vector of the PREVIOUS step alone (sos at t = 0) -- no
                   context, the probe's built-in floor
        layer l  = output of global layer l, l = 1..12
    <out>/meta.npz            labels and bookkeeping, identical for every model
                              (the windows are chosen by --seed, not by model)

Windows start at a random bar of the span, not only on a I bar, so a bar's
phase in the cycle is not a function of its position in the window. The
position-only control in probe_train measures how much is left.

Models, each given as NAME=SPEC:
    base=/path/to/v0.42.ckpt               a checkpoint file
    finetune=$WORK/checkpoints_v6/cp_bass_ft_v6_s1
                                           a run directory -> its best checkpoint
    random=random:0                        random init, torch seed 0

    python -m midi_adapter.probe_extract \\
        --data_prefix $WORK/data/v6_chordal --out_dir $WORK/probe_r1 \\
        --model base=$BASE_CKPT \\
        --model finetune=$WORK/checkpoints_v6/cp_bass_ft_v6_s1 \\
        --model random=random:0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from models.chord_tracr_rule_model import ChordTracrRuleModel, N_CHORDS

BAR = 16                      # subbeats in a 4/4 bar
CYCLE = 4                     # bars in I-IV-V-I
ROMAN_OF_PHASE = [0, 1, 2, 0]  # I IV V I  ->  classes I=0, IV=1, V=2
ROMAN_NAMES = ['I', 'IV', 'V']
SPLITS = ('finetune', 'unseen')


# ---------------------------------------------------------------------------
# Spans and windows
# ---------------------------------------------------------------------------

def load_split(prefix: str, split: str):
    path = f'{prefix}_{split}.pt'
    data = torch.load(path, weights_only=True)
    length = torch.load(f'{prefix}_{split}.length.pt', weights_only=True)
    chords = torch.load(f'{prefix}_{split}.chords.pt', weights_only=True)
    files = []
    with open(f'{prefix}_{split}.txt') as f:
        for line in f:
            files.append(line.split('\t')[1].split('@')[0])
    assert len(files) == len(length) == len(chords), path
    return data, length, chords, files


def choose_windows(prefix: str, window_bars: int, per_span: int,
                   max_spans: int, seed: int):
    """[(split, span index, file, tonic, start bar, raw window), ...]."""
    rng = np.random.default_rng(seed)
    out = []
    for split in SPLITS:
        if not os.path.exists(f'{prefix}_{split}.pt'):
            print(f'  {split}: no file, skipped')
            continue
        data, length, chords, files = load_split(prefix, split)
        start = (torch.cumsum(length, 0) - length).tolist()
        idx = np.arange(len(length))
        if max_spans and len(idx) > max_spans:
            idx = np.sort(rng.choice(idx, max_spans, replace=False))
        n_before = len(out)
        for i in idx:
            n_bars = int(length[i]) // BAR
            starts = np.arange(0, n_bars - window_bars + 1)
            if len(starts) == 0:
                continue
            if len(starts) > per_span:
                starts = np.sort(rng.choice(starts, per_span, replace=False))
            for s in starts:
                a = start[i] + int(s) * BAR
                out.append((split, int(i), files[i], int(chords[i]), int(s),
                            data[a:a + window_bars * BAR]))
        print(f'  {split}: {len(idx)} spans -> {len(out) - n_before} windows')
    return out


def pc_set(step: torch.Tensor) -> int:
    """12-bit pitch-class set of one raw CP step (max_polyphony x 4 uint8)."""
    d = step.view(-1, 4).long()
    ok = (d[:, 1] != 255) & (d[:, 0] != 254) & (d[:, 0] != 127)
    m = 0
    for p in d[ok, 1].tolist():
        m |= 1 << (p % 12)
    return m


def build_meta(windows, window_bars: int, rule: ChordTracrRuleModel):
    rows = {k: [] for k in ('split', 'span', 'src_file', 'tonic', 'start_bar',
                            'bar', 'pos', 'phase', 'roman', 'next_chord',
                            'prev_pcset', 'ft_split')}
    for split, i, f, tonic, s, win in windows:
        for b in range(1, window_bars):
            phase = (s + b) % CYCLE
            rows['split'].append(split)
            rows['span'].append(i)
            rows['src_file'].append(f)
            rows['tonic'].append(tonic)
            rows['start_bar'].append(s)
            rows['bar'].append(b)
            rows['pos'].append(b * BAR)
            rows['phase'].append(phase)
            rows['roman'].append(ROMAN_OF_PHASE[phase])
            rows['next_chord'].append(int(rule.cadence[tonic, phase]))
            # what the model heard last: the control task's input
            rows['prev_pcset'].append(pc_set(win[b * BAR - 1]))
            # FramedDataset's split of this span: r > 1 is what a model
            # trained on the finetune file saw, r in {0, 1} it never did
            rows['ft_split'].append(i % 10)
    return {k: np.asarray(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_net(spec: str, model_size: int) -> RoFormerSymbolicTransformer:
    if spec.startswith('random:'):
        torch.manual_seed(int(spec.split(':', 1)[1]))
        print('    random init')
        return RoFormerSymbolicTransformer(size=model_size)
    path = spec
    if os.path.isdir(spec):
        from midi_adapter.cp_results_table import best_ckpt
        path = best_ckpt(spec)
        if path is None:
            raise SystemExit(f'no monitored checkpoint in {spec}')
    net = RoFormerSymbolicTransformer(size=model_size)
    st = torch.load(path, map_location='cpu')
    net.load_state_dict(st.get('state_dict', st), strict=True)
    print(f'    {path}')
    return net


@torch.no_grad()
def hidden_states(net, raw: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """(B, len(positions), n_layers + 1, hidden) at the given output positions.

    Mirrors RoFormerSymbolicTransformer.forward up to the global encoder:
    output position t sees sos + steps 0..t-1 and predicts step t.
    """
    B, T, _ = raw.shape
    x = net.preprocess(raw.long(), torch.zeros(B, dtype=torch.long, device=raw.device))
    h, _ = net.local_encode(x)
    h = h.view(B, T, -1)
    sos = net.global_sos.view(1, 1, -1).expand(B, 1, -1)
    h = torch.cat([sos, h[:, :-1]], dim=1)
    out = net.model(h, attention_mask=net.buffered_future_mask(h),
                    output_hidden_states=True, return_dict=True)
    hs = torch.stack(out.hidden_states, dim=2)          # (B, T, L+1, H)
    return hs[:, positions]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--data_prefix', required=True,
                   help='e.g. $WORK/data/v6_chordal; reads _finetune and _unseen')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model', action='append', required=True,
                   help='NAME=SPEC; SPEC is a .ckpt, a run directory, or random:SEED')
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--window_bars', type=int, default=8,
                   help='8 bars = 128 steps, the v6 TRAIN_LENGTH')
    p.add_argument('--windows_per_span', type=int, default=2)
    p.add_argument('--max_spans', type=int, default=0,
                   help='per split; 0 = all. A smoke run uses e.g. 100')
    p.add_argument('--chord_minor', default='harmonic')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    rule = ChordTracrRuleModel(rule_mode='period4', minor_pattern=args.chord_minor)

    print(f'windows of {args.window_bars} bars from {args.data_prefix}')
    windows = choose_windows(args.data_prefix, args.window_bars,
                             args.windows_per_span, args.max_spans, args.seed)
    meta = build_meta(windows, args.window_bars, rule)
    np.savez(os.path.join(args.out_dir, 'meta.npz'), **meta,
             window_bars=args.window_bars, n_chords=N_CHORDS)
    n_per = args.window_bars - 1
    N = len(windows) * n_per
    print(f'{N} probe examples ({len(windows)} windows x {n_per} bars)')
    positions = [b * BAR for b in range(1, args.window_bars)]

    for item in args.model:
        name, spec = item.split('=', 1)
        print(f'  {name}:')
        net = load_net(spec, args.model_size).to(device).eval()
        L1, H = net.num_layers + 1, net.hidden_size
        d = os.path.join(args.out_dir, name)
        os.makedirs(d, exist_ok=True)
        feats = np.lib.format.open_memmap(os.path.join(d, 'feats.npy'), mode='w+',
                                          dtype=np.float16, shape=(N, L1, H))
        for a in range(0, len(windows), args.batch_size):
            raw = torch.stack([w[-1] for w in windows[a:a + args.batch_size]]).to(device)
            hs = hidden_states(net, raw, positions)        # (B, n_per, L1, H)
            feats[a * n_per:(a + len(raw)) * n_per] = (
                hs.reshape(-1, L1, H).float().cpu().numpy().astype(np.float16))
        feats.flush()
        with open(os.path.join(d, 'source.txt'), 'w') as f:
            f.write(spec + '\n')
        print(f'    wrote {d}/feats.npy  {feats.shape}')
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
