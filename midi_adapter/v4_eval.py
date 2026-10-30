"""
The v4 results table: real-data prompts, scored on downbeats.

v2/v3 prompted every model with a bar drawn from the synthetic generator.
v4's models are trained on real bass lines selected from the LA MIDI corpus
(prepare_v4_data), and the synthetic generator cannot write a prompt in their
representation: 16 subbeats to the bar at polyphony 16, with real rhythm. So
the prompts here are the opening bar(s) of real spans, taken from the VAL
split of each data file -- the same songs FramedDataset holds out from
training (index % 10 == 1), so no finetuned model saw them.

Every span starts on a I bar with the root on its downbeat, so the key is in
the prompt's first slot, exactly where the rule model reads it.

Three readings of each generated bar, from strictest to loosest:

  downbeat        the v2/v3 metric: the note struck on the downbeat IS the
                  rule's root. One slot, one note.
  root_exact      the bar's root, inferred from a duration-weighted
                  pitch-class histogram of everything sounding in the bar
                  (triad template match: root + fifth + the stronger third),
                  equals the rule's root. Real bass lines walk, so a bar can
                  state its chord without striking the root first.
  root_chordtone  the inferred root is the root, third or fifth of the rule's
                  chord: the bar is in the right harmony, possibly inverted.

plus the harmonic profile of every sounding step (chord / scale / outside
relative to the rule's root, and onsets per bar). The **Data** row runs the
real continuation of each prompt through the same scorers. Downbeat is 1.000
there by construction; root_exact is NOT, and its value is the ceiling of the
histogram method on real music -- the number a model should be read against.

Key groups are two-way. The host pretrained on the whole LA corpus, so every
key is seen in pretraining; the finetune files hold ten keys and F# and G#
are held out of finetuning entirely (prepare_v4_data --holdout_keys 6 8).

Model loading is cp_results_table.build_entries, untouched, so the rule model
is rebuilt from each checkpoint's recorded configuration exactly as before.

Usage
-----
  python -m midi_adapter.v4_eval \\
      --base_ckpt <v0.42 ckpt> --ft_ckpt <finetune ckpt> \\
      --ckpt_dir checkpoints_v4 --runs cp_yy_frozen_v4_s1 ... \\
      --prompt_data data/v4_bass_finetune.pt data/v4_bass_unseen.pt \\
                    data/v4_bass_pretrain.pt \\
      --n_prompt_beats 16 --n_gen_beats 112 --n_trials 10 \\
      --save_midi_dir eval_midi_v4/s1 --json_out logs_v4/table_s1.json

  --data_only needs no checkpoint: it prints the Data row and writes the
  data MIDIs, which is how the prompt pool is checked before any training.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.evaluate_cp_bass import (
    UNSEEN, HARMONY_KEYS, CHORD_STEPS, SCALE_STEPS,
)
from midi_adapter.infer_cp_bass import _sample, decode_output
from midi_adapter.generate_synthetic_bass import (
    OFFSETS, ROOT_NAMES, DURATION_TEMPLATES,
    SECONDS_PER_SUBBEAT as _SYNTH_SECONDS_PER_SUBBEAT,
)

# beat_div=4 and 120 BPM, as the listening samples: one subbeat is 0.125 s.
# decode_output times notes with the synthetic generator's constant (0.5 s,
# one subbeat per beat there), so its output is rescaled by this ratio.
SECONDS_PER_SUBBEAT = 0.125
_MIDI_TIME_SCALE = SECONDS_PER_SUBBEAT / _SYNTH_SECONDS_PER_SUBBEAT


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_prompt_pool(paths: list[str], window: int, split_ratio: int = 10,
                     split: str = 'val') -> dict[int, list[tuple[str, int, torch.Tensor]]]:
    """{key: [(file, song index, raw window (window, subseq) uint8), ...]}.

    Songs are taken from the VAL split exactly as FramedDataset defines it
    (index % split_ratio == 1), in file order, so the pool is deterministic
    and disjoint from what any model trained on from that file. A song
    shorter than `window` is skipped: the Data row needs the real continuation.
    """
    pool: dict[int, list] = collections.defaultdict(list)
    for p in paths:
        data = torch.load(p, weights_only=True)
        length = torch.load(p[:-3] + '.length.pt', weights_only=True)
        keys = torch.load(p[:-3] + '.keys.pt', weights_only=True)
        start = torch.cumsum(length, 0) - length
        for i in range(len(length)):
            r = i % split_ratio
            in_split = (split == 'all' or (split == 'val' and r == 1)
                        or (split == 'test' and r == 0) or (split == 'train' and r > 1))
            if not in_split or int(length[i]) < window:
                continue
            s = int(start[i])
            pool[int(keys[i])].append((os.path.basename(p), i, data[s:s + window]))
    return pool


def pick_prompts(pool, key: int, n: int):
    """The first n val-split spans of a key. Fewer exist -> all of them."""
    return pool.get(key, [])[:n]


# ---------------------------------------------------------------------------
# Scoring (vectorised over a batch of continuations)
# ---------------------------------------------------------------------------

def infer_root(hist: torch.Tensor) -> torch.Tensor:
    """(..., 12) pitch-class weights -> (...) root by triad template.

    score[r] = w[r] + 0.5 * w[r+7] + 0.5 * max(w[r+3], w[r+4]): the root
    itself counts full, the fifth and whichever third is present count half,
    so a bar that walks root-third-fifth and one that sits on the root both
    name the same chord, while a bar sitting on the fifth names the fifth.
    """
    r = torch.arange(12, device=hist.device)
    fifth = hist[..., (r + 7) % 12]
    third = torch.maximum(hist[..., (r + 3) % 12], hist[..., (r + 4) % 12])
    return (hist + 0.5 * fifth + 0.5 * third).argmax(-1)


METRICS = ('acc', 'root_exact', 'root_chordtone', 'bass_exact', 'bass_chordtone')
METRIC_NAMES = {'acc': 'downbeat slot-0 note is the rule root',
                'root_exact': 'histogram root (all notes) equals the rule root',
                'root_chordtone': 'histogram root (all notes) is root, 3rd or 5th of the rule chord',
                'bass_exact': 'histogram root of the bass line (lowest note per step) equals the rule root',
                'bass_chordtone': 'bass-line histogram root is root, 3rd or 5th of the rule chord'}


def score_tokens(gen: torch.Tensor, keys: torch.Tensor, n_prompt: int, spc: int):
    """gen: (B, n_gen, subseq) PROCESSED tokens for absolute steps
    n_prompt .. n_prompt+n_gen-1. Returns per-sample downbeat accuracy and
    the harmonic profile, both as (B,) tensors in a dict.

    Same reading as the in-training probe and evaluate_cp_bass._extract_pc:
    slot 0 is the program token, slot 1 is pitch + (dur+1)*128; a step whose
    first slot is not a note (eos/pad) counts as WRONG on a downbeat and as
    silence elsewhere.
    """
    B, n_gen, _ = gen.shape
    prog, pd = gen[..., 0], gen[..., 1]
    # program 127 is the drum kit: its 'pitch' is a kit piece, not a pitch
    # class, so drums never enter a histogram.
    valid = (prog < 127) & (pd >= 128)                      # (B, n_gen)
    pc = (pd % 128) % 12
    t = torch.arange(n_prompt, n_prompt + n_gen, device=gen.device)
    off = torch.tensor(OFFSETS, device=gen.device)[(t // spc) % 4]
    exp = (keys.view(-1, 1) + off.view(1, -1)) % 12         # (B, n_gen)
    down = (t % spc == 0)
    acc = ((pc == exp) & valid)[:, down].float().mean(1)

    # Histogram root per bar. Every note slot of the step counts (a
    # generated step may hold several), weighted by its duration in steps,
    # clipped at the bar line so a long note does not vote for the next bar.
    W = gen.shape[-1] // 2
    pairs = gen.view(B, n_gen, W, 2)
    v_all = (pairs[..., 0] < 127) & (pairs[..., 1] >= 128)          # (B, n_gen, W)
    pc_all = (pairs[..., 1] % 128) % 12
    dur_idx = (pairs[..., 1] // 128 - 1).clamp(0, len(DURATION_TEMPLATES) - 1)
    dur = torch.tensor(DURATION_TEMPLATES, device=gen.device)[dur_idx].float()
    remaining = (spc - (t % spc)).float().view(1, n_gen, 1)
    w = torch.minimum(dur, remaining) * v_all
    n_bars_i = n_gen // spc
    bar_of = ((t - n_prompt) // spc).view(1, n_gen, 1).expand(B, n_gen, W)
    hist = torch.zeros(B, n_bars_i, 12, device=gen.device)
    flat_idx = (bar_of * 12 + pc_all).reshape(B, -1)
    hist = hist.view(B, -1).scatter_add(1, flat_idx, w.reshape(B, -1)).view(B, n_bars_i, 12)
    gt = (keys.view(-1, 1) + torch.tensor(OFFSETS, device=gen.device)[
        (torch.arange(n_bars_i, device=gen.device) + n_prompt // spc) % 4].view(1, -1)) % 12

    def tiers(h):
        rel = (infer_root(h) - gt) % 12
        sounding = h.sum(-1) > 0
        return (((rel == 0) & sounding).float().mean(1),
                (((rel == 0) | (rel == 3) | (rel == 4) | (rel == 7)) & sounding)
                .float().mean(1))

    root_exact, root_chordtone = tiers(hist)

    # The bass line: the lowest sounding note of each step (v5 has chords
    # above the bass; in v4 every step holds one note and this equals the
    # all-notes histogram). Same duration weighting and bar clipping.
    pitch_all = pairs[..., 1] % 128
    lowest = torch.where(v_all, pitch_all, torch.full_like(pitch_all, 999)).argmin(-1)   # (B, n_gen)
    is_low = torch.zeros_like(v_all).scatter_(2, lowest.unsqueeze(-1), True) & v_all
    hist_b = torch.zeros(B, n_bars_i * 12, device=gen.device).scatter_add(
        1, flat_idx, (w * is_low).reshape(B, -1)).view(B, n_bars_i, 12)
    bass_exact, bass_chordtone = tiers(hist_b)

    step = (pc - exp) % 12
    in_chord = torch.zeros(12, dtype=torch.bool, device=gen.device)
    in_chord[list(CHORD_STEPS)] = True
    in_scale = torch.zeros(12, dtype=torch.bool, device=gen.device)
    in_scale[list(SCALE_STEPS)] = True
    onsets = valid.sum(1).clamp(min=1).float()
    n_bars = n_gen / spc
    return {
        'acc':            acc,
        'root_exact':     root_exact,
        'root_chordtone': root_chordtone,
        'bass_exact':     bass_exact,
        'bass_chordtone': bass_chordtone,
        'chord':          (in_chord[step] & valid).sum(1) / onsets,
        'scale':          (in_scale[step] & valid).sum(1) / onsets,
        'outside':        (~in_chord[step] & ~in_scale[step] & valid).sum(1) / onsets,
        'onsets_per_bar': valid.sum(1).float() / n_bars,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, is_adapter: bool, prompt: torch.Tensor, total: int,
             temperature: float, device) -> torch.Tensor:
    """(B, total - n_prompt, subseq) processed tokens of the continuation."""
    n_prompt, width = prompt.shape[1], prompt.shape[2]
    # The prompt is a [:, :n_prompt] slice of the processed window, which is
    # not contiguous, and local_encode .view()s its input.
    prompt = prompt.contiguous()
    if is_adapter:
        sampled = model.global_sampling(prompt, max_seq_len=total,
                                        temperature=temperature)
        return torch.stack(sampled[n_prompt:], 1)
    # _sample is B=1, and its steps stop at EOS, so they come back at
    # different widths; pad them out to the data's width like the prompt.
    pad = model.tokenizer.pad_token
    outs = []
    for i in range(prompt.shape[0]):
        s = _sample(model, prompt[i:i + 1], total, temperature, device,
                    show_progress=False)
        steps = [_fit_width(t, width, pad) for t in s[n_prompt:]]
        outs.append(torch.cat(steps, 0))
    return torch.stack(outs, 0)


def _fit_width(step: torch.Tensor, width: int, pad: int) -> torch.Tensor:
    """(1, w) -> (1, width): right-padded with pad, or cut to width."""
    w = step.shape[-1]
    if w >= width:
        return step[..., :width]
    return torch.cat([step, torch.full((step.shape[0], width - w), pad,
                                       dtype=step.dtype, device=step.device)], -1)


def write_midi(tokens: torch.Tensor, tokenizer, path: str) -> None:
    """tokens: (T, subseq) processed tokens -> MIDI at 120 BPM, beat_div 4."""
    pm = decode_output([tokens[t:t + 1].cpu() for t in range(tokens.shape[0])],
                       tokenizer)
    for ins in pm.instruments:
        for n in ins.notes:
            n.start *= _MIDI_TIME_SCALE
            n.end *= _MIDI_TIME_SCALE
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pm.write(path)


# ---------------------------------------------------------------------------
# One model over the prompt pool
# ---------------------------------------------------------------------------

def eval_entry(label, model, is_adapter, base, pool, args, device, midi_dir):
    """{key: {'acc':..., 'chord':..., ...}} averaged over that key's prompts.

    model=None scores the DATA: the real continuation of each prompt, run
    through the same scorer -- the reference row.
    """
    spc = args.beats_per_chord
    total = args.n_prompt_beats + args.n_gen_beats
    per_key = {}
    for key in range(12):
        items = pick_prompts(pool, key, args.n_trials)
        if not items:
            continue
        x = torch.stack([w for _, _, w in items]).to(device)          # (B, total, 64)
        proc = base.preprocess(x, torch.zeros(len(items), dtype=torch.long,
                                              device=device))
        keys = torch.full((len(items),), key, device=device)
        if model is None:
            gen = proc[:, args.n_prompt_beats:]
        else:
            gen = generate(model, is_adapter, proc[:, :args.n_prompt_beats],
                           total, args.temperature, device)
        sc = score_tokens(gen, keys, args.n_prompt_beats, spc)
        per_key[key] = {k: float(v.mean()) for k, v in sc.items()}
        per_key[key]['n'] = len(items)
        if midi_dir:
            # Prompt + continuation of the first trial, so the file opens on
            # the bar the model was given and what it made of it follows.
            full = torch.cat([proc[0, :args.n_prompt_beats], gen[0]], 0)
            write_midi(full, base.tokenizer,
                       os.path.join(midi_dir, label, f'key_{ROOT_NAMES[key]}.mid'))
    return per_key


def row_means(per_key: dict, keys: list[int]) -> dict:
    ks = [k for k in keys if k in per_key]
    if not ks:
        return {m: float('nan') for m in METRICS + HARMONY_KEYS}
    return {m: float(np.mean([per_key[k][m] for k in ks]))
            for m in METRICS + HARMONY_KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--base_ckpt', default=None, help='pretrained host (Pretrain row)')
    p.add_argument('--ft_ckpt', default=None, help='finetuned host, no adapter')
    p.add_argument('--ckpt_dir', default='checkpoints_v4')
    p.add_argument('--runs', nargs='+', default=[], help='run directory names')
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip', type=int, default=1)
    p.add_argument('--rule_key_select', default='first', choices=['phase0', 'first'])
    p.add_argument('--prompt_data', nargs='+', required=True,
                   help='v4 .pt files; prompts come from their VAL split')
    p.add_argument('--prompt_split', default='val',
                   choices=['val', 'test', 'train', 'all'])
    p.add_argument('--beats_per_chord', type=int, default=16,
                   help='subbeats per bar in the data (16 for v4)')
    p.add_argument('--n_prompt_beats', type=int, default=16)
    p.add_argument('--n_gen_beats', type=int, default=112)
    p.add_argument('--n_trials', type=int, default=10,
                   help='prompts per key (fewer if the split has fewer)')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--save_midi_dir', default=None)
    p.add_argument('--json_out', default=None)
    p.add_argument('--data_only', action='store_true',
                   help='no checkpoints: the Data row and data MIDIs only')
    p.add_argument('--skip_pretrain', action='store_true',
                   help='leave out the Pretrain row (a partial table whose '
                        'Pretrain row comes from the full one at merge time)')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.n_prompt_beats % args.beats_per_chord:
        sys.exit(f'--n_prompt_beats {args.n_prompt_beats} is not whole bars '
                 f'(beats_per_chord {args.beats_per_chord}); the first generated '
                 f'step would not be a downbeat.')
    if not args.data_only and not args.base_ckpt:
        sys.exit('--base_ckpt is required unless --data_only')

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    total = args.n_prompt_beats + args.n_gen_beats
    seen = [k for k in range(12) if k not in UNSEEN]
    rows = [('Finetuned keys', seen, '{' + ','.join(map(str, seen)) + '}'),
            ('Unseen', UNSEEN, '{' + ','.join(map(str, UNSEEN)) + '}'),
            ('Overall', list(range(12)), 'all 12 keys')]

    print('=' * 60)
    print('Prompts')
    print('=' * 60)
    pool = load_prompt_pool(args.prompt_data, total, split=args.prompt_split)
    print(f'  {args.prompt_split} split of {len(args.prompt_data)} file(s); '
          f'{args.n_prompt_beats}-step prompt, {args.n_gen_beats} generated, '
          f'temperature {args.temperature}')
    for key in range(12):
        n = len(pool.get(key, []))
        use = min(n, args.n_trials)
        flag = '' if use == args.n_trials else f'   <-- only {use}'
        print(f'  {ROOT_NAMES[key]:3} (key {key:2d}): {n:4d} spans, using {use}{flag}')

    print()
    print('=' * 60)
    print('Loading models')
    print('=' * 60)
    if args.data_only:
        # Tokenisation only (preprocess has no weights); the host is never run.
        base = RoFormerSymbolicTransformer(size=args.model_size, max_lr=1e-4,
                                           with_velocity=False).to(device).eval()
        entries = []
        print('  --data_only: no checkpoints loaded')
    else:
        from midi_adapter.cp_results_table import build_entries
        entries = build_entries(args, device)
        base = entries[0][1]
        if args.skip_pretrain:
            entries = [e for e in entries if e[0] != 'Pretrain']

    print()
    print('=' * 60)
    print('Evaluating')
    print('=' * 60)
    results, metrics, chord, per_key_all = {}, {m: {} for m in METRICS}, {}, {}
    for lbl, model, is_ad in [('Data', None, False)] + list(entries):
        print(f'  {lbl} ...', end='  ', flush=True)
        pk = eval_entry(lbl, model, is_ad, base, pool, args, device,
                        args.save_midi_dir)
        per_key_all[lbl] = pk
        results[lbl], chord[lbl] = {}, {}
        for m in METRICS:
            metrics[m][lbl] = {}
        for rlbl, keys, _ in rows:
            mean = row_means(pk, keys)
            results[lbl][rlbl] = mean['acc']
            for m in METRICS:
                metrics[m][lbl][rlbl] = mean[m]
            chord[lbl][rlbl] = {k: mean[k] for k in HARMONY_KEYS}
        print('  '.join(f'{metrics["root_exact"][lbl][r]:.3f}' for r, _, _ in rows)
              + '   (root_exact)')

    protocol = {'n_prompt_beats': args.n_prompt_beats,
                'n_gen_beats': args.n_gen_beats, 'n_trials': args.n_trials,
                'beats_per_chord': args.beats_per_chord,
                'temperature': args.temperature,
                'prompt_data': args.prompt_data, 'prompt_split': args.prompt_split}
    print_tables(protocol, rows, results, metrics, chord, per_key_all)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump({'protocol': protocol,
                       # 'results' stays the downbeat metric so aggregate_seeds
                       # reads it as before; the three tiers are under 'metrics'.
                       'results': results, 'metrics': metrics, 'chord_tone': chord,
                       'rows': [[lbl, keys, s] for lbl, keys, s in rows],
                       'per_key': {l: {str(k): v for k, v in pk.items()}
                                   for l, pk in per_key_all.items()}},
                      f, indent=2)
        print(f'\nwrote {args.json_out}')
    if args.save_midi_dir:
        print(f'midi -> {args.save_midi_dir}/<model>/key_<root>.mid '
              f'(prompt first, then the continuation; Data/ is the real one)')


def print_tables(protocol, rows, results, metrics, chord, per_key_all):
    """The three tiers, the harmonic profile and the per-key line, as text.
    Shared with merge_tables so a merged table prints exactly like a direct one."""
    labels = list(results)
    spc = protocol['beats_per_chord']
    print()
    print(f'{protocol["n_prompt_beats"]}-step prompt, {protocol["n_gen_beats"]} generated, '
          f'{protocol["n_gen_beats"] // spc} scored bars per prompt, '
          f'temperature {protocol["temperature"]}')
    for m in METRICS:
        print()
        print(f'{m}: {METRIC_NAMES[m]}')
        print(f'{"":<16}' + ''.join(f'{l[:18]:>20}' for l in labels))
        for rlbl, _, s in rows:
            print(f'{rlbl:<16}' + ''.join(f'{metrics[m][l][rlbl]:>20.3f}' for l in labels)
                  + f'   {s}')
    print()
    print('harmonic profile of every sounding step (Unseen row; Overall in JSON)')
    print(f'{"":<16}' + ''.join(f'{m:>16}' for m in HARMONY_KEYS))
    for l in labels:
        print(f'{l[:16]:<16}' + ''.join(f'{chord[l]["Unseen"][m]:>16.3f}'
                                        for m in HARMONY_KEYS))
    print()
    print('per key, root_exact')
    print(f'{"":<16}' + ''.join(f'{ROOT_NAMES[k]:>6}' for k in range(12)))
    for l in labels:
        pk = {int(k): v for k, v in per_key_all[l].items()}
        print(f'{l[:16]:<16}' + ''.join(
            f'{pk[k]["root_exact"]:>6.2f}' if k in pk else f'{"-":>6}'
            for k in range(12)))


if __name__ == '__main__':
    main()
