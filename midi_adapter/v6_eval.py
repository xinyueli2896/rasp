"""
v6 evaluation: does the generation follow the CHORD cadence?

v4 and v5 scored roots, which cannot distinguish a major cadence from the
minor one on the same tonic -- their root sequences are identical. v6's rule is
over the 24 triads, so the headline here is an exact chord match, and the root
and quality numbers are the diagnostic split of it:

    chord_acc   the generated bar's chord IS the cadence's chord
    root_acc    its root is right                 (v4/v5's question)
    qual_acc    its third is right                (what v4/v5 could not see)

A model that reads the tonic correctly and then holds major produces
root_acc 1.000 and chord_acc 0.25 on a minor cadence, which is exactly the
failure v4/v5's metrics read as perfect.

Three hypotheses are scored on every generation, so a failure is named rather
than guessed:

    rule        CADENCE[tonic chord]                  -- the result
    parallel    CADENCE[same root, other mode]        -- "right tonic, wrong
                mode": identical roots, so v4/v5 would call this a success
    relative    CADENCE[relative major or minor]      -- drift to the host's
                tonal prior, the "Am as vi of C major" reading. Its roots are
                disjoint from the rule's, so this one also craters root_acc

Chords are read from a generated bar the way the whole pipeline reads them:
the root from slot 0 of the bar's downbeat, the quality from the
duration-weighted major-against-minor third over the rest of the bar. Same
definition as census_v6_quality, prepare_v6_data and the adapter's own
_tonic_chord, so nothing can drift between selection, conditioning and
scoring.

    # no checkpoints: the Data and Pretrain rows only, which is also the
    # scorer's self-test -- Data MUST come out at chord_acc 1.000, because
    # prepare_v6_data selected those spans for matching the cadence
    python -m midi_adapter.v6_eval --data_prefix data/v6_chordal --data_only

    python -m midi_adapter.v6_eval --data_prefix data/v6_chordal \\
        --base_ckpt <v0.42>.ckpt --ckpt_dir checkpoints_v6 \\
        --runs cp_yy_frozen_v6_s1 cp_yy_lora_v6_s1
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.v4_eval import generate, write_midi
# Already solved there: train_cp_yinyang names its series by whichever
# metric it monitored, so the best checkpoint is the MAX unseen_acc or the
# MIN val_loss -- not the alphabetically first, and never last.ckpt, which
# carries no metric at all.
from midi_adapter.cp_results_table import best_ckpt
from midi_adapter.generate_synthetic_bass import DURATION_TEMPLATES
from midi_adapter.evaluate_cp_yinyang import _report_load
from models.chord_tracr_rule_model import (
    ChordTracrRuleModel, CHORD_NAMES, N_CHORDS, N_POS, N_ROOTS, chord_index,
    chord_root, chord_is_minor,
)

METRICS = ('chord_acc', 'root_acc', 'qual_acc')
HYPOTHESES = ('rule', 'parallel', 'relative')


def corrupt(tonic: torch.Tensor, how: str, rng: torch.Generator) -> torch.Tensor:
    """A deliberately WRONG tonic chord to hand the rule model.

    The intervention v6 turns on. The adapter reads its tonic chord from the
    prompt, so feeding it a different one asks the question no accuracy number
    answers on its own: is the cadence channel load-bearing, or is the host
    doing this from the prompt alone? The v5 analysis makes that the sharp
    question -- those adapters already produced harmonic minor at 0.97-0.99
    with no quality channel at all, matching the corpus's own 0.981.

      follows the corrupted cadence  -> the channel drives the output
      follows the true one           -> the channel is decoration

    'parallel' keeps the root and flips the mode, which is the subtlest lie:
    the roots it implies are identical, so only the thirds can reveal which
    cadence was followed. 'relative' and 'shift' change the roots too, so they
    show up in root_acc as well.
    """
    if how == 'parallel':
        return (tonic + N_CHORDS // 2) % N_CHORDS
    if how == 'relative':
        return torch.tensor([relative_of(int(c)) for c in tonic],
                            device=tonic.device)
    if how == 'shift':
        # Same mode, root up a tritone: maximally far in root space while
        # leaving the mode untouched, so root_acc alone separates them.
        return (tonic + 6) % N_ROOTS + (tonic // N_ROOTS) * N_ROOTS
    if how == 'random':
        return torch.randint(0, N_CHORDS, tonic.shape, generator=rng,
                             device=tonic.device)
    raise ValueError(f'unknown --corrupt_tonic {how!r}')


def parallel_of(tonic: int) -> int:
    """Same root, other mode -- identical roots, so v4/v5 cannot tell them apart."""
    return (tonic + N_CHORDS // 2) % N_CHORDS


def relative_of(tonic: int) -> int:
    """A major tonic's relative minor, or a minor tonic's relative major."""
    r = chord_root(tonic)
    return (chord_index((r - 3) % 12, True) if not chord_is_minor(tonic)
            else chord_index((r + 3) % 12, False))


# ---------------------------------------------------------------------------
# Prompt pool, keyed on the tonic CHORD
# ---------------------------------------------------------------------------

def load_chord_pool(paths: list[str], window: int, split_ratio: int = 10,
                    split: str = 'val'):
    """{tonic chord: [(file, song index, raw window), ...]}.

    The val split as FramedDataset defines it (index % split_ratio == 1), in
    file order, so the pool is deterministic and disjoint from what a model
    trained on that file saw. A song shorter than `window` is skipped: the
    Data row needs a real continuation to score.
    """
    pool: dict[int, list] = collections.defaultdict(list)
    for p in paths:
        data = torch.load(p, weights_only=True)
        length = torch.load(p[:-3] + '.length.pt', weights_only=True)
        chords = torch.load(p[:-3] + '.chords.pt', weights_only=True)
        start = torch.cumsum(length, 0) - length
        for i in range(len(length)):
            r = i % split_ratio
            keep = (split == 'all' or (split == 'val' and r == 1)
                    or (split == 'test' and r == 0)
                    or (split == 'train' and r > 1))
            if not keep or int(length[i]) < window:
                continue
            s = int(start[i])
            pool[int(chords[i])].append((os.path.basename(p), i, data[s:s + window]))
    return pool


# ---------------------------------------------------------------------------
# Reading chords out of generated tokens
# ---------------------------------------------------------------------------

def read_bar_chords(gen: torch.Tensor, spc: int):
    """(roots, is_minor, voiced) each (B, n_bars), from processed tokens.

    gen : (B, T, subseq) processed tokens, T a whole number of bars.
    root    pitch class in slot 0 of the bar's downbeat -- the bass, which is
            where prepare_v6_data puts it and what its invariant checks
    quality duration-weighted mass on the major third against the minor,
            relative to that root, over the bar with slot 0 excluded
    voiced  False where neither third sounds, so a bar with no third is not
            silently counted as major
    """
    B, T, W2 = gen.shape
    n_bars = T // spc
    pairs = gen[:, :n_bars * spc, :].reshape(B, n_bars, spc, W2 // 2, 2)
    valid = (pairs[..., 0] < 127) & (pairs[..., 1] >= 128)
    pitch = pairs[..., 1] % 128
    roots = (pitch[:, :, 0, 0] % 12).clamp(0, 11)              # (B, n_bars)

    body = valid.clone()
    body[:, :, :, 0] = False                                   # drop the bass
    dur_idx = (pairs[..., 1] // 128 - 1).clamp(0, len(DURATION_TEMPLATES) - 1)
    dur = torch.as_tensor(DURATION_TEMPLATES, device=gen.device,
                          dtype=torch.float32)[dur_idx]
    t = torch.arange(spc, device=gen.device).view(1, 1, spc, 1)
    w = torch.minimum(dur, (spc - t).float()).clamp(min=0) * body
    rel = (pitch - roots[:, :, None, None]) % 12
    maj = (w * (rel == 4)).sum((2, 3))
    mino = (w * (rel == 3)).sum((2, 3))
    return roots, mino > maj, (maj + mino) > 0


def score(gen: torch.Tensor, tonic: torch.Tensor, rule: ChordTracrRuleModel,
          n_prompt: int, spc: int, extra: dict | None = None) -> dict:
    """Per-sample scores for each hypothesis, plus a per-bar-index breakdown.

    `extra` adds named hypotheses keyed on their own tonic chords -- used by
    --corrupt_tonic to score the generation against the cadence the rule model
    was actually FED, beside the one the data says it should follow.
    """
    roots, is_min, voiced = read_bar_chords(gen, spc)
    B, n_bars = roots.shape
    got = roots + 12 * is_min.long()
    bar0 = n_prompt // spc                       # the first GENERATED bar
    phases = (torch.arange(n_bars, device=gen.device) + bar0) % N_POS
    cad = rule.cadence.to(gen.device)

    out: dict = {}
    hyps = [('rule', tonic),
            ('parallel', torch.tensor([parallel_of(int(c)) for c in tonic],
                                      device=gen.device)),
            ('relative', torch.tensor([relative_of(int(c)) for c in tonic],
                                      device=gen.device))]
    hyps += list((extra or {}).items())
    for name, idx in hyps:
        want = cad[idx][:, phases]                           # (B, n_bars)
        chord_hit = (got == want) & voiced
        root_hit = roots == (want % 12)
        qual_hit = (is_min == (want >= 12)) & voiced
        out[f'{name}/chord_acc'] = chord_hit.float().mean(1)
        out[f'{name}/root_acc'] = root_hit.float().mean(1)
        out[f'{name}/qual_acc'] = qual_hit.float().mean(1)
        if name == 'rule':
            out['_per_bar'] = chord_hit.float()              # (B, n_bars)
            out['_phases'] = phases
    out['unvoiced'] = (~voiced).float().mean(1)
    return out


# ---------------------------------------------------------------------------

def eval_entry(label, model, is_adapter, base, pool, rule, args, device,
               per_bar_acc, midi_dir=None, corrupt_how=None):
    """{tonic chord: {metric: value}} averaged over that chord's prompts."""
    spc = args.beats_per_chord
    total = args.n_prompt_beats + args.n_gen_beats
    out = {}
    for tonic, items in sorted(pool.items()):
        if not items:
            continue
        win = torch.stack([w for _, _, w in items]).to(device)
        proc = base.preprocess(win.long(), torch.zeros(len(items), dtype=torch.long,
                                                       device=device))
        prompt = proc[:, :args.n_prompt_beats]
        tn = torch.full((len(items),), tonic, device=device, dtype=torch.long)
        extra = None
        if corrupt_how and is_adapter:
            # Override what the adapter reads off the prompt, so the rule model
            # is fed a cadence the music does not follow. Patched on the
            # instance rather than in cp_yinyang: the adapter's own extractor
            # stays the one that trained, and this lie exists only here.
            fed = corrupt(tn, corrupt_how, torch.Generator(device=device).manual_seed(0))
            model._tonic_chord = types.MethodType(
                lambda self, _x, _f=fed: _f, model)
            extra = {'fed': fed}
        if model is None:                       # the DATA row: the real continuation
            gen = proc[:, args.n_prompt_beats:total]
        else:
            gen = generate(model, is_adapter, prompt, total, args.temperature, device)
        sc = score(gen, tn, rule, args.n_prompt_beats, spc, extra)
        out[tonic] = {k: float(v.mean()) for k, v in sc.items()
                      if not k.startswith('_')}
        out[tonic]['n'] = len(items)
        pb = sc['_per_bar'].mean(0)
        for i, v in enumerate(pb.tolist()):
            per_bar_acc[label][i].append(v)
        if midi_dir:
            d = os.path.join(midi_dir, label)
            os.makedirs(d, exist_ok=True)
            both = torch.cat([prompt[:1], gen[:1]], dim=1)
            write_midi(both[0], base.tokenizer,
                       os.path.join(d, f'chord_{CHORD_NAMES[tonic]}.mid'))
    return out


# Config buffers recorded by CPYinyangTransformer. Absent from older
# checkpoints, in which case the command line supplies the value.
_CONFIG_BUFFERS = ('rule_subbeats_per_chord', 'rule_key_select_first',
                   'rule_rate_bar', 'strict_causal')


def load_v6_adapter(ck: str, args) -> CPYinyangTransformer:
    """Rebuild a v6 adapter exactly as it trained, then load its weights.

    Two silent failures this replaces. (1) train_cp_yinyang's Lightning wrapper
    holds the adapter as `self.model`, so every key in the .ckpt starts with
    `model.`; loading that with strict=False into a bare CPYinyangTransformer
    matched NOTHING and left a randomly initialised host and adapter, which is
    what the first v6 tables scored. (2) The rebuild took lora_rank,
    rule_input, rule_rate and rule_key_select from defaults rather than from
    the run, so LoRA and the noinput encoder were dropped even when the keys
    lined up. Everything recoverable is now read from the checkpoint, and a
    load that leaves any real weight missing stops the eval.
    """
    raw = torch.load(ck, map_location='cpu')
    sd = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    if any(k.startswith('model.') for k in sd):
        sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}

    def buf(name):
        return next((v for k, v in sd.items() if k.endswith(name)), None)

    n_layers = [6, 12, 24, 32][args.model_size]
    n_adapters = len({int(m.group(1)) for k in sd
                      for m in [re.search(r'yinyang_attn\.(\d+)\.', k)] if m})
    n_skip = n_layers // n_adapters if n_adapters else args.n_skip
    lora_rank = next((v.shape[0] for k, v in sd.items() if 'lora_A' in k), 0)
    rule_input = ('ar_state' if any('rule_input_encoder.fc_out' in k for k in sd)
                  else 'tokens')
    w1 = next((v for k, v in sd.items()
               if k.endswith('rule_input_encoder.fc1.weight')), None)
    enc_type = 'mlp' if (w1 is not None or rule_input == 'tokens') else 'linear'
    enc_hidden = w1.shape[0] if w1 is not None else 64
    spc, rr, ks, sc = (buf(n) for n in _CONFIG_BUFFERS)

    cfg = dict(
        adapter_rank=args.adapter_rank, n_skip=n_skip, lora_rank=lora_rank,
        rule_mode='chord', rule_view=args.rule_view,
        rule_input=rule_input, rule_enc_type=enc_type, rule_enc_hidden=enc_hidden,
        rule_subbeats_per_chord=int(spc) if spc is not None else args.beats_per_chord,
        rule_rate=('bar' if bool(rr) else 'beat') if rr is not None else 'bar',
        rule_key_select=('first' if bool(ks) else 'phase0') if ks is not None else 'first',
        strict_causal=bool(sc) if sc is not None else False,
        chord_minor_pattern=args.minor_pattern)

    base = RoFormerSymbolicTransformer(size=args.model_size)
    # The pretrained host first, so an adapter-only state dict still sits on
    # the right host; a full Lightning checkpoint then overwrites it.
    if args.base_ckpt and os.path.exists(args.base_ckpt):
        bst = torch.load(args.base_ckpt, map_location='cpu')
        base.load_state_dict(bst.get('state_dict', bst))
    ad = CPYinyangTransformer(base_model=base, **cfg)
    missing, unexpected = ad.load_state_dict(sd, strict=False)
    _report_load(missing, unexpected)
    real_missing = [k for k in missing
                    if not k.startswith('base.')
                    and not any(k.endswith(c) for c in _CONFIG_BUFFERS)]
    if real_missing or unexpected:
        raise SystemExit(f'{ck}: rebuild does not match the checkpoint -- '
                         f'{len(real_missing)} adapter weights missing '
                         f'({real_missing[:3]}), {len(unexpected)} unexpected '
                         f'({unexpected[:3]}). Refusing to score a model that '
                         f'did not load.')
    print(f'    loaded {len(sd)} tensors  n_skip={n_skip} lora_rank={lora_rank} '
          f'rule_input={rule_input} spc={cfg["rule_subbeats_per_chord"]} '
          f'rate={cfg["rule_rate"]} key_select={cfg["rule_key_select"]}')
    return ad


def row(per_chord: dict, chords: list[int], metric: str) -> float:
    vals = [(per_chord[c][metric], per_chord[c]['n'])
            for c in chords if c in per_chord]
    n = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / n if n else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--data_prefix', required=True,
                   help='e.g. data/v6_chordal; reads _finetune and _unseen')
    p.add_argument('--base_ckpt', default=None)
    p.add_argument('--ckpt_dir', default=None)
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--data_only', action='store_true',
                   help='no checkpoints: the Data row alone, which is the '
                        "scorer's self-test -- it must read chord_acc 1.000")
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--adapter_rank', type=int, default=256)
    p.add_argument('--n_skip', type=int, default=1)
    p.add_argument('--beats_per_chord', type=int, default=16)
    p.add_argument('--n_prompt_beats', type=int, default=16)
    p.add_argument('--n_gen_beats', type=int, default=112)
    p.add_argument('--n_trials', type=int, default=5)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--minor_pattern', default='harmonic',
                   choices=['harmonic', 'natural'])
    p.add_argument('--rule_view', default='root_next')
    p.add_argument('--split', default='val',
                   choices=['val', 'test', 'train', 'all'])
    p.add_argument('--corrupt_tonic', default=None,
                   choices=['parallel', 'relative', 'shift', 'random'],
                   help='the intervention: feed the rule model a WRONG tonic '
                        'chord and report whether the generation follows the '
                        'cadence it was fed or the one the music follows. '
                        "parallel keeps the root and flips the mode, so only "
                        'the thirds can tell them apart')
    p.add_argument('--save_midi_dir', default=None)
    p.add_argument('--json_out', default=None)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total = args.n_prompt_beats + args.n_gen_beats
    rule = ChordTracrRuleModel(minor_pattern=args.minor_pattern)

    ft = f'{args.data_prefix}_finetune.pt'
    us = f'{args.data_prefix}_unseen.pt'
    pool_ft = load_chord_pool([ft], total, split=args.split) if os.path.exists(ft) else {}
    pool_us = load_chord_pool([us], total, split='all') if os.path.exists(us) else {}
    # The held-out chords are never trained on, so every one of their spans is
    # fair game as a prompt; the finetune chords use the val split only.
    pool = {**pool_ft, **pool_us}
    pool = {c: v[:args.n_trials] for c, v in pool.items() if v}
    seen = sorted(pool_ft)
    unseen = sorted(pool_us)
    if not pool:
        sys.exit(f'no prompts: {ft} / {us} missing, or no span reaches '
                 f'{total} subbeats')
    print(f'prompts: {sum(len(v) for v in pool.values())} over '
          f'{len(pool)} tonic chords  (finetune {len(seen)}, unseen {len(unseen)})')
    print(f'protocol: {args.n_prompt_beats}-subbeat prompt + {args.n_gen_beats} '
          f'generated, bar = {args.beats_per_chord}, T = {args.temperature}')
    print(f'unseen chords: {" ".join(CHORD_NAMES[c] for c in unseen) or "none"}')

    base = RoFormerSymbolicTransformer(size=args.model_size)
    if args.base_ckpt and os.path.exists(args.base_ckpt):
        st = torch.load(args.base_ckpt, map_location='cpu')
        base.load_state_dict(st.get('state_dict', st))
        print(f'host: {args.base_ckpt}')
    elif not args.data_only:
        print('WARNING: no --base_ckpt; the host is randomly initialised')
    base = base.to(device).eval()

    entries = [('Data', None, False)]
    if not args.data_only:
        entries.append(('Pretrain', base, False))
        for run in args.runs:
            ck = best_ckpt(os.path.join(args.ckpt_dir, run)) if args.ckpt_dir else None
            if ck is None:
                print(f'  {run}: no monitored checkpoint in '
                      f'{os.path.join(args.ckpt_dir or ".", run)} -- skipped')
                continue
            ad = load_v6_adapter(ck, args)
            entries.append((run, ad.to(device).eval(), True))
            print(f'  {run}: {os.path.basename(ck)}')

    per_chord_all, per_bar_acc = {}, collections.defaultdict(
        lambda: collections.defaultdict(list))
    with torch.no_grad():
        for label, model, is_ad in entries:
            print(f'  {label} ...', end='  ', flush=True)
            per_chord_all[label] = eval_entry(
                label, model, is_ad, base, pool, rule, args, device,
                per_bar_acc, args.save_midi_dir, args.corrupt_tonic)
            print('done')

    print('\n' + '=' * 78)
    print('the rule hypothesis')
    hdr = f'{"model":<26}' + ''.join(f'{m:>13}' for m in METRICS) + f'{"unvoiced":>11}'
    print(hdr)
    for label in per_chord_all:
        pc = per_chord_all[label]
        cells = ''.join(f'{row(pc, sorted(pc), "rule/" + m):>13.3f}' for m in METRICS)
        print(f'{label:<26}{cells}{row(pc, sorted(pc), "unvoiced"):>11.3f}')

    print('\nfinetune chords vs held-out chords, on chord_acc')
    print(f'{"model":<26}{"finetune":>12}{"unseen":>12}'
          f'{"unseen major":>14}{"unseen minor":>14}')
    for label, pc in per_chord_all.items():
        um = [c for c in unseen if not chord_is_minor(c)]
        un = [c for c in unseen if chord_is_minor(c)]
        print(f'{label:<26}{row(pc, seen, "rule/chord_acc"):>12.3f}'
              f'{row(pc, unseen, "rule/chord_acc"):>12.3f}'
              f'{row(pc, um, "rule/chord_acc"):>14.3f}'
              f'{row(pc, un, "rule/chord_acc"):>14.3f}')

    print('\nwhich cadence did it actually follow? (chord_acc per hypothesis)')
    print(f'{"model":<26}' + ''.join(f'{h:>12}' for h in HYPOTHESES))
    for label, pc in per_chord_all.items():
        print(f'{label:<26}' + ''.join(
            f'{row(pc, sorted(pc), h + "/chord_acc"):>12.3f}' for h in HYPOTHESES))
    print('  parallel high with rule low = read the tonic, defaulted to the')
    print('  other mode -- the failure v4/v5 metrics score as perfect.')

    if args.corrupt_tonic:
        print(f'\nINTERVENTION: the rule model was fed a {args.corrupt_tonic} '
              'tonic chord, not the one the music follows')
        print(f'{"model":<26}{"follows FED":>13}{"follows TRUE":>14}'
              f'{"fed root":>11}{"fed qual":>10}')
        for label, pc in per_chord_all.items():
            if label == 'Data':
                continue        # no model, so nothing was fed anything
            print(f'{label:<26}{row(pc, sorted(pc), "fed/chord_acc"):>13.3f}'
                  f'{row(pc, sorted(pc), "rule/chord_acc"):>14.3f}'
                  f'{row(pc, sorted(pc), "fed/root_acc"):>11.3f}'
                  f'{row(pc, sorted(pc), "fed/qual_acc"):>10.3f}')
        print('  FED high, TRUE low  -> the cadence channel drives the output')
        print('  TRUE high, FED low  -> the channel is ignored; the prompt alone')
        print('                         is doing the work, as v5 suggested it might')

    print('\nchord_acc by generated bar index (cycle bar in brackets)')
    bars = sorted(next(iter(per_bar_acc.values())))
    print(f'{"model":<26}' + ''.join(
        f'{f"b{b + 1}[{(b + 1 + args.n_prompt_beats // args.beats_per_chord - 1) % N_POS}]":>9}'
        for b in bars))
    for label in per_chord_all:
        print(f'{label:<26}' + ''.join(
            f'{sum(per_bar_acc[label][b]) / len(per_bar_acc[label][b]):>9.3f}'
            for b in bars))

    print('\nper tonic chord, chord_acc, DATA row only -- the scorer self-test  (* held out)')
    for mino in (False, True):
        for half in (range(6), range(6, 12)):
            cells = []
            for r in half:
                c = chord_index(r, mino)
                v = per_chord_all[next(iter(per_chord_all))].get(c)
                tag = '*' if c in unseen else ' '
                cells.append(f'{CHORD_NAMES[c]:>4}{tag}'
                             + (f'{v["rule/chord_acc"]:>6.2f}' if v else '     -'))
            print('  ' + '  '.join(cells))

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump({'protocol': vars(args),
                       'unseen': [CHORD_NAMES[c] for c in unseen],
                       'per_chord': {l: {CHORD_NAMES[c]: v for c, v in pc.items()}
                                     for l, pc in per_chord_all.items()},
                       'per_bar': {l: {str(b): sum(v) / len(v)
                                       for b, v in d.items()}
                                   for l, d in per_bar_acc.items()}}, f, indent=2)
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
