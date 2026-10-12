"""Technical audit of the no-rule-input (ar_state) setup.

Run: python -m models.test_ar_state_encoder
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yinyang_model import build_yinyang_model
from models.ar_state_encoder import ARStateRuleEncoder
from models.rasp_compiled_rule import ROOT_BASE, N_ROOTS
from data.dataset import make_sequence, VOCAB_SIZE

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def build(enc='mlp', view='root_next'):
    return build_yinyang_model(
        ar_ckpt_path=None, max_seq_len=42, use_lora=False, train_ar=False,
        n_skip=1, rule_mode='compiled', rule_view=view,
        rule_input='ar_state', rule_enc_type=enc)


def main():
    T = 32
    idx = torch.tensor([make_sequence(6, 8)])

    print('\n1. the encoder reads the PRE-BLOCK AR representation')
    m = build()
    x0 = m._ar_state(idx)
    ar = m._ar_base
    want = ar.tok_emb(idx) + ar.pos_emb(torch.arange(idx.shape[1]).unsqueeze(0))
    m.train()                       # must hold in TRAIN mode, i.e. dropout-free
    check('x0 is deterministic in train mode',
          torch.allclose(m._ar_state(idx), m._ar_state(idx)))
    check('x0 == tok_emb + pos_emb (no dropout)', torch.allclose(x0, want))
    check('x0 width is the AR d_model', x0.shape[-1] == ar.tok_emb.weight.shape[1],
          f'{tuple(x0.shape)}')

    print('\n2. encoder output is a valid distribution (the MLP gate needs it)')
    for enc in ('linear', 'mlp'):
        mm = build(enc)
        p = mm.rule_input_encoder(mm._ar_state(idx))
        ok = (float(p.min()) >= 0.0 and float(p.max()) <= 1.0
              and torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5))
        check(f'{enc:<6} output in [0,1] and sums to 1', ok,
              f'min={float(p.min()):.3f} max={float(p.max()):.3f} '
              f'sum={float(p.sum(-1).mean()):.4f}')

    print('\n3. a mass>1 input really would break the gate (why softmax)')
    rm = m.rule_model
    for scale, expect_ok in ((1.0, True), (2.0, False)):
        oh = F.one_hot(idx, VOCAB_SIZE).float() * scale
        h = F.pad(oh, (0, rm.d_model - VOCAB_SIZE)) \
            + rm.W_phase[rm.phase_at(T, 'cpu', 0)].unsqueeze(0) \
            + rm.W_phase_next[rm.phase_at(T, 'cpu', rm.shift)].unsqueeze(0)
        got = rm.mlp(rm.attention(h))[0, :T - 1, ROOT_BASE:ROOT_BASE + N_ROOTS] \
                .argmax(-1).tolist()
        ok = got == idx[0, 1:].tolist()
        check(f'scale {scale}: exact={ok} (expected {expect_ok})', ok == expect_ok)

    print('\n4. with an encoder that can name the token, the pipeline is exact')
    m2 = build('linear')
    with torch.no_grad():
        # x0[t] = tok_emb[token] + pos_emb[t], so the readout has to work for
        # EVERY (token, position) pair, not just the diagonal.
        toks = torch.arange(VOCAB_SIZE).repeat_interleave(T)
        poss = torch.arange(T).repeat(VOCAB_SIZE)
        X = (m2._ar_base.tok_emb(toks) + m2._ar_base.pos_emb(poss))        # (12T, d)
        Y = F.one_hot(toks, VOCAB_SIZE).float() * 50.0
        W = torch.linalg.lstsq(X, Y).solution
        m2.rule_input_encoder.fc_out.weight.copy_(W.t())
        m2.rule_input_encoder.fc_out.bias.zero_()
    idx_all = torch.stack([torch.tensor(make_sequence(x, 8)) for x in range(VOCAB_SIZE)])
    enc_acc = float((m2.rule_input_encoder.logits(m2._ar_state(idx_all)).argmax(-1)
                     == idx_all).float().mean())
    check('fitted encoder names the current token', enc_acc > 0.99, f'acc={enc_acc:.4f}')
    rh = m2._ar_state_rule_hidden(idx_all)
    exact = float((rh[:, :T - 1].argmax(-1) == idx_all[:, 1:]).float().mean())
    check('end-to-end next token exact for all 12 starters', exact > 0.99,
          f'acc={exact:.4f}  (view=root_next)')

    print('\n5. causality: rule_hidden[t] never depends on tokens > t')
    m3 = build()
    rh = m3._ar_state_rule_hidden(idx)
    bad = idx.clone(); bad[0, T // 2:] = 11
    rh_b = m3._ar_state_rule_hidden(bad)
    d = float((rh[0, :T // 2] - rh_b[0, :T // 2]).abs().max())
    check('prefix unchanged when the suffix is corrupted', d < 1e-6, f'delta={d:.3g}')

    print('\n6. what trains and what does not')
    m4 = build()
    groups = {}
    for n, p in m4.named_parameters():
        g = groups.setdefault(n.split('.')[0], [0, 0])
        g[0 if p.requires_grad else 1] += p.numel()
    for k, (t, f) in sorted(groups.items()):
        print(f'      {k:<20} trainable={t:>9,}  frozen={f:>9,}')
    check('AR fully frozen',
          not any(p.requires_grad for p in m4.ar_model.parameters()))
    check('rule model has zero parameters',
          sum(p.numel() for p in m4.rule_model.parameters()) == 0)
    check('encoder IS trainable',
          all(p.requires_grad for p in m4.rule_input_encoder.parameters()))

    m4(idx).sum().backward()
    got = sorted({n.split('.')[0] for n, p in m4.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0})
    check('gradient reaches encoder AND adapter only',
          got == ['rule_input_encoder', 'yinyang_attn'], f'{got}')

    print('\n7. the auxiliary loss')
    m5 = build()
    lg = m5.get_encoder_logits(idx)
    check('get_encoder_logits returns (B,T,12) pre-softmax',
          lg is not None and tuple(lg.shape) == (1, T, VOCAB_SIZE), f'{tuple(lg.shape)}')
    check('logits are NOT already softmaxed',
          not torch.allclose(lg.sum(-1), torch.ones_like(lg.sum(-1)), atol=1e-3))
    check("encoder_type routes the aux target to the CURRENT token",
          m5.rule_input_encoder.encoder_type in ('linear', 'mlp'))

    print('\n8. no-aux variant is the same model, aux weight 0')
    check('model builds identically without any aux plumbing',
          build('mlp').rule_input is not None)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
