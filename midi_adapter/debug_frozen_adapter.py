"""
Why does the frozen-AR adapter free-run far below what it was trained to do?

Separates three numbers that the single rule-following accuracy conflates:

  teacher-forced   the training computation exactly -- every position gets its
                   adapter correction, inputs are ground truth. Upper bound on
                   what the trained adapter knows.

  free-run ALL     autoregressive, but the correction is applied to EVERY
                   position of the recomputed sequence, so the host's context
                   matches training. Isolates the cost of generating its own
                   inputs.

  free-run LAST    autoregressive with the correction applied only at the last
                   position -- what CPYinyangTransformer.global_sampling does
                   today. Isolates the cost of the train/inference mismatch.

The mismatch is real: the training forward (cp_yinyang.py:473) runs

    h = h + yinyang_attn[k](h, rule_h, sub_offset=0)        # ALL positions

while global_sampling (cp_yinyang.py:545) runs

    correction = yinyang_attn[k](h_out[:, -1:, :], rule_h, sub_offset=i)
    h_out = cat([h_out[:, :-1, :], h_out[:, -1:, :] + correction], dim=1)

and each step recomputes all global layers from `h`, which accumulates only
local-encoded beats -- never block outputs. So an earlier position's correction
is neither persisted nor recomputed, and the host's self-attention at position i
reads the UNCORRECTED base representation of every earlier beat. In the frozen
stem the correction is the only thing that changes the host's behaviour, so
dropping it from the context leaves position i attending to something that looks
like the plain pretrain base -- which scores 0.000 on unseen keys.

A large ALL-vs-LAST gap means the deployed sampler, not the adapter, is what the
Unseen accuracy is measuring.

Usage
-----
  python -m midi_adapter.debug_frozen_adapter \\
      --base_ckpt    ckpt/cp_bass_pre_A_aligned/....ckpt \\
      --adapter_ckpt checkpoints_cp/cp_yy_frozen/....ckpt \\
      --n_prompt_beats 1 --n_gen_beats 23
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.evaluate_cp_bass import (CATEGORIES, ALL_SEEN, UNSEEN,
                                           _expected_pc, _scored_positions)
from midi_adapter.evaluate_cp_yinyang import load_model
from midi_adapter.infer_cp_bass import _prompt_from_key


# ---------------------------------------------------------------------------
# rule_hidden, exactly as global_sampling builds it
# ---------------------------------------------------------------------------

def _rule_hidden_for_sampling(model: CPYinyangTransformer, x: torch.Tensor,
                              max_seq_len: int) -> torch.Tensor | None:
    if model.bidirectional:
        return None
    pc_0 = (x[:, 0, 1] % 128 % 12).clamp(0, 11)
    return model.rule_model.build_rule_hidden_analytic(pc_0, max_seq_len, x.device)


# ---------------------------------------------------------------------------
# Free-running generation, with the correction site under our control
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model: CPYinyangTransformer, x: torch.Tensor, max_seq_len: int,
             correct_all: bool) -> list[torch.Tensor]:
    """global_sampling, but `correct_all` chooses who gets the correction.

    correct_all=False reproduces the shipped sampler exactly.
    correct_all=True  reproduces the training forward's correction pattern.
    """
    base = model.base
    batch_size, seed_len, subseq_len = x.shape
    rule_hidden = _rule_hidden_for_sampling(model, x, max_seq_len)

    h_seed, _ = base.local_encode(x)
    h_seed = h_seed.view(batch_size, seed_len, base.hidden_size)
    sos = base.global_sos.view(1, 1, -1).expand(batch_size, 1, -1)
    h = torch.cat([sos, h_seed], dim=1)

    y = [x[:, i, :] for i in range(seed_len)]

    for i in range(seed_len, max_seq_len):
        attn_mask = base.buffered_future_mask(h)
        sinusoidal_pos = base.model.embed_positions(h.shape[:-1], 0)[None, None, :, :]

        h_out = h
        for j, layer in enumerate(base.model.layer):
            h_out = layer(h_out, attention_mask=attn_mask,
                          sinusoidal_pos=sinusoidal_pos)[0]
            if (j + 1) % model.n_skip != 0:
                continue
            k = (j + 1) // model.n_skip - 1
            rule_h = (model.ar_to_rule[k](h_out) if model.bidirectional
                      else rule_hidden)
            if correct_all:
                # Whole sequence, absolute beat 0 at index 0 -- identical to
                # the training forward.
                h_out = h_out + model.yinyang_attn[k](h_out, rule_h, sub_offset=0)
            else:
                corr = model.yinyang_attn[k](h_out[:, -1:, :], rule_h, sub_offset=i)
                h_out = torch.cat([h_out[:, :-1, :], h_out[:, -1:, :] + corr], dim=1)

        y_next = base.local_sampling(h_out[:, -1], max_subseq_len=subseq_len,
                                     temperature=0)
        y.append(y_next)
        h_enc = base.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
        h = torch.cat([h, h_enc], dim=1)

    return y


# ---------------------------------------------------------------------------
# The three measurements
# ---------------------------------------------------------------------------

def _pc(tok_row: torch.Tensor) -> int:
    """Pitch class from a preprocessed CP beat: slot 1 = pitch + (dur+1)*128."""
    return int(tok_row[1] % 128 % 12)


def _prompt(key, n_beats, device, model, cfg):
    """max_polyphony and cycle_unit must match the training data, or the
    prompt is the wrong token width and local_encode does not mask the pad."""
    return _prompt_from_key(key, n_prompt_bars=0, device=device,
                            base=model.base, n_prompt_beats=n_beats,
                            max_polyphony=cfg['max_polyphony'],
                            cycle_unit=cfg['cycle_unit'])


@torch.no_grad()
def _teacher_forced_logits(model, x, rule_hidden=None):
    """CPYinyangTransformer.forward, with rule_hidden optionally supplied.

    Mirrors cp_yinyang.py:508 exactly. The override exists so the same host
    input can be run against a DIFFERENT key's rule signal, which is what
    isolates "the adapter is ignoring the rule" from "the adapter is using it".
    """
    base = model.base
    B, T, _ = x.shape

    h, emb = base.local_encode(x)
    h = h.view(B, T, base.hidden_size)
    sos = base.global_sos.view(1, 1, -1).expand(B, 1, -1)
    h = torch.cat([sos, h[:, :-1]], dim=1)
    model._pre_block_cache = h

    if rule_hidden is None:
        if model.rule_input == 'ar_state':
            rule_hidden = model._ar_state_rule_hidden(h)
        elif model.bidirectional:
            rule_hidden = None
        else:
            rule_hidden = model._rule_hidden(x)

    mask = base.buffered_future_mask(h)
    for i, layer in enumerate(base.model.layer):
        h = layer(h, attention_mask=mask)[0]
        if (i + 1) % model.n_skip == 0:
            k = (i + 1) // model.n_skip - 1
            rule_h = model.ar_to_rule[k](h) if model.bidirectional else rule_hidden
            h = h + model.yinyang_attn[k](h, rule_h, sub_offset=0)
    return base.local_decode(h, emb)


def _hits(pred_pc, key, n_prompt, total, bpc):
    """Accuracy over the beats that carry a rule target."""
    pos = _scored_positions(n_prompt, total, bpc)
    return [int(pred_pc(t) == _expected_pc(key, t, bpc)) for t in pos]


@torch.no_grad()
def acc_teacher_forced(model, key, n_prompt, n_gen, device, cfg) -> float:
    """The training computation: ground-truth inputs, all positions corrected."""
    total = n_prompt + n_gen
    x = _prompt(key, total, device, model, cfg)
    B, T, S = x.shape
    # local_encode flattens (B, T, subseq) -> (B*T, subseq), so local_decode
    # returns (B*T, subseq, vocab) rather than (B, T, subseq, vocab).
    logits = _teacher_forced_logits(model, x)
    assert logits.shape[:2] == (B * T, S), \
        f'unexpected logits shape {tuple(logits.shape)}; expected ({B*T}, {S}, vocab)'
    pred = logits.argmax(-1).view(B, T, S)
    return float(np.mean(_hits(lambda t: _pc(pred[0, t]), key, n_prompt,
                               total, cfg['beats_per_chord'])))


@torch.no_grad()
def acc_free_run(model, key, n_prompt, n_gen, device, correct_all, cfg) -> float:
    total = n_prompt + n_gen
    prompt = _prompt(key, n_prompt, device, model, cfg)
    y = generate(model, prompt, total, correct_all=correct_all)
    return float(np.mean(_hits(lambda t: _pc(y[t][0]), key, n_prompt,
                               total, cfg['beats_per_chord'])))


@torch.no_grad()
def rule_sensitivity(model, key, n_prompt, n_gen, device, cfg):
    """Does the output depend on the rule input at all?

    The gates are open in every bar-experiment run, including the ones whose
    accuracy equals the bare pretrained host's. An open gate only means the
    branch has magnitude -- it says nothing about whether what flows through
    carries the key. So: hold the host input fixed and swap the rule model's
    input to a tritone-away key. If nothing downstream moves, the adapter is
    injecting a large signal that is constant in the key, and the host is
    right to ignore it.

    Returns (acc_true, acc_wrong, frac_pred_changed, rel_logit_delta).
    """
    total = n_prompt + n_gen
    bpc = cfg['beats_per_chord']
    x = _prompt(key, total, device, model, cfg)
    # A tritone away: maximally far in pitch-class space, and never equal to
    # the true key, so any invariance is not an accident of the offsets.
    x_other = _prompt((key + 6) % 12, total, device, model, cfg)

    rh_true = model._rule_hidden(x)
    rh_wrong = model._rule_hidden(x_other)

    lg_true = _teacher_forced_logits(model, x, rh_true)
    lg_wrong = _teacher_forced_logits(model, x, rh_wrong)

    B, T, S = x.shape
    p_true = lg_true.argmax(-1).view(B, T, S)
    p_wrong = lg_wrong.argmax(-1).view(B, T, S)

    a_true = float(np.mean(_hits(lambda t: _pc(p_true[0, t]), key,
                                 n_prompt, total, bpc)))
    a_wrong = float(np.mean(_hits(lambda t: _pc(p_wrong[0, t]), key,
                                  n_prompt, total, bpc)))

    pos = _scored_positions(n_prompt, total, bpc)
    changed = float(np.mean([int(_pc(p_true[0, t]) != _pc(p_wrong[0, t]))
                             for t in pos]))
    denom = lg_true.float().norm()
    rel = float((lg_true.float() - lg_wrong.float()).norm() / denom) if denom > 0 else 0.0
    return a_true, a_wrong, changed, rel


def main():
    p = argparse.ArgumentParser(
        description='Separate adapter quality from the sampler mismatch.')
    p.add_argument('--base_ckpt',    required=True)
    p.add_argument('--adapter_ckpt', required=True)
    p.add_argument('--model_size',     type=int, default=1)
    p.add_argument('--adapter_rank',   type=int, default=256)
    p.add_argument('--n_skip',         type=int, default=4)
    p.add_argument('--rule_mode',      type=str, default='compiled')
    p.add_argument('--rule_view',      type=str, default='root_next')
    p.add_argument('--lora_rank',      type=int, default=0)
    p.add_argument('--rule_input',     type=str, default='tokens')
    p.add_argument('--rule_enc_type',  type=str, default='mlp')
    p.add_argument('--rule_enc_hidden',type=int, default=64)
    p.add_argument('--n_prompt_beats', type=int, default=1)
    p.add_argument('--n_gen_beats',    type=int, default=23)
    p.add_argument('--device',         type=str, default=None)
    # The latent-rule (one chord per bar) configuration. These MUST match the
    # data and the training command. beats_per_chord drives both the expected
    # pitch class and which beats are scored; rule_key_select and
    # rule_subbeats_per_chord rebuild the rule model -- getting the last one
    # wrong is what silently invalidated an entire round of result tables.
    p.add_argument('--beats_per_chord', type=int, default=1)
    p.add_argument('--rule_key_select', type=str, default='phase0',
                   choices=['phase0', 'first'])
    p.add_argument('--rule_subbeats_per_chord', type=int, default=None,
                   help='Defaults to --beats_per_chord, which is what every '
                        'bar run trained with.')
    p.add_argument('--max_polyphony',  type=int, default=4)
    p.add_argument('--strict_causal',  action='store_true')
    # _prompt_from_key calls generate_song, which draws a random bar pattern
    # per call, so one trial per key resolves to 1/7 per scored downbeat and
    # the Overall row disagrees with the group rows. Average like the results
    # table does. Free-running is the expensive part: cost is roughly linear
    # in n_trials.
    p.add_argument('--n_trials',       type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    spc = (args.rule_subbeats_per_chord if args.rule_subbeats_per_chord
           is not None else args.beats_per_chord)
    cfg = {'beats_per_chord': args.beats_per_chord,
           'max_polyphony': args.max_polyphony,
           'cycle_unit': 'bar' if args.beats_per_chord > 1 else 'beat'}

    model = load_model(args.base_ckpt, args.adapter_ckpt, args.model_size,
                       args.adapter_rank, args.n_skip, False, False,
                       'embedding', args.rule_mode, device,
                       rule_view       = args.rule_view,
                       lora_rank       = args.lora_rank,
                       rule_input      = args.rule_input,
                       rule_enc_type   = args.rule_enc_type,
                       rule_enc_hidden = args.rule_enc_hidden,
                       strict_causal   = args.strict_causal,
                       rule_subbeats_per_chord = spc,
                       rule_key_select = args.rule_key_select)
    model.eval()

    rows = list(CATEGORIES) + [('Overall', ALL_SEEN + UNSEEN, 'all 12 keys')]
    cols = [('teacher-forced', lambda k: acc_teacher_forced(
                model, k, args.n_prompt_beats, args.n_gen_beats, device, cfg)),
            ('free-run ALL',   lambda k: acc_free_run(
                model, k, args.n_prompt_beats, args.n_gen_beats, device, True, cfg)),
            ('free-run LAST',  lambda k: acc_free_run(
                model, k, args.n_prompt_beats, args.n_gen_beats, device, False, cfg))]

    print('=' * 78)
    print(f'FROZEN-ADAPTER DIAGNOSIS  ({args.n_prompt_beats}-beat prompt, '
          f'{args.n_gen_beats} generated, greedy)')
    print(f'  base    {args.base_ckpt}')
    print(f'  adapter {args.adapter_ckpt}')
    print(f'  rule    spc={spc} key_select={args.rule_key_select} '
          f'bpc={args.beats_per_chord} mp={args.max_polyphony} '
          f'cycle={cfg["cycle_unit"]}')
    if args.beats_per_chord > 1:
        print('          scoring downbeats only -- off-beats may use chord tones')
    print('=' * 78)
    print('  free-run LAST = the shipped sampler.  free-run ALL = corrections at')
    print('  every position, matching the training forward.')
    print()
    hdr = f'{"Key group":<16} {"Keys":<22}' + ''.join(f' {c:>15}' for c, _ in cols)
    print(hdr); print('-' * len(hdr))

    res = {}
    for lbl, keys, kstr in rows:
        line = f'{lbl:<16} {kstr:<22}'
        for cname, fn in cols:
            v = float(np.mean([fn(k) for k in keys
                               for _ in range(args.n_trials)]))
            res[(lbl, cname)] = v
            line += f' {v:>15.3f}'
        print(line)
    print('=' * len(hdr))

    # --- is the rule input used at all? ------------------------------------
    # Only meaningful when something external feeds the rule model. With
    # rule_input='ar_state' the signal is derived from the host's own state,
    # so there is no other key's input to swap in.
    if model.rule_input == 'tokens' and not model.bidirectional:
        print()
        print('Rule-input sensitivity: same host input, rule model fed the '
              'tritone-away key.')
        print('  An open gate carrying no key information looks like '
              'acc_true == acc_wrong, changed == 0.')
        print()
        sh = (f'{"Key group":<16} {"Keys":<22} {"acc true":>9} {"acc wrong":>10} '
              f'{"changed":>8} {"d logits":>9}')
        print(sh); print('-' * len(sh))
        for lbl, keys, kstr in rows:
            vals = [rule_sensitivity(model, k, args.n_prompt_beats,
                                     args.n_gen_beats, device, cfg)
                    for k in keys for _ in range(args.n_trials)]
            at, aw, ch, rl = (float(np.mean([v[i] for v in vals])) for i in range(4))
            print(f'{lbl:<16} {kstr:<22} {at:>9.3f} {aw:>10.3f} '
                  f'{ch:>8.3f} {rl:>9.4f}')
        print('-' * len(sh))

    gap = res[('Unseen', 'free-run ALL')] - res[('Unseen', 'free-run LAST')]
    exposure = res[('Unseen', 'teacher-forced')] - res[('Unseen', 'free-run ALL')]
    print()
    print(f'Unseen: ALL - LAST  = {gap:+.3f}   (size of the OLD sampler bug;')
    print('                                 global_sampling corrects every')
    print('                                 position now, so ALL is what ships')
    print('                                 and the results table matches it)')
    print(f'Unseen: TF  - ALL   = {exposure:+.3f}   (cost of generating its own inputs)')
    print()
    if res[('Unseen', 'teacher-forced')] < 0.5:
        print('  The adapter did not learn to route the rule on unseen keys.')
        print('  Teacher-forced is the ceiling and it is already low, so no')
        print('  change to generation can help -- check the rule-input')
        print('  sensitivity block above before anything else.')
    elif exposure > 0.2:
        print('  The adapter KNOWS the rule (high teacher-forced) and loses it')
        print('  while free-running. Exposure bias, not rule learning: the')
        print('  deficit is robustness to its own generations, and the fix is')
        print('  on the training/decoding side, not the adapter.')
    else:
        print('  The adapter knows the rule and keeps it while free-running.')
        print('  This stem is working as intended.')


if __name__ == '__main__':
    main()
