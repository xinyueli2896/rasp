"""
X11 round 1, step 2: linear probes for the cadence, with their controls.

Reads what probe_extract wrote and fits, for every model, task and layer, a
multinomial logistic regression from the 768-d hidden state to the label:

    z = W x + c                      W: K x 768, x z-scored per layer
    L = -1/N sum_i w[y_i] log softmax(z_i)[y_i]  +  lam * ||W||^2

w[k] is the inverse frequency of class k in the training set, so a common
class cannot win by default. Fitted full batch with L-BFGS; the loss is convex,
so there is one optimum and no dependence on initialisation. lam is chosen per
(model, task, layer) by 5-fold cross-validation over training FILES, on
validation cross-entropy, and then held fixed across split seeds.

Tasks (both read just before bar b's downbeat):
    roman       3 classes  I, IV, V        -- the cadence's function, key-free
    next_chord  24 classes                 -- function bound to the key

Splits, per seed: files of the 20 trained tonic chords go 80/20 into probe
train/test. The four held-out tonic chords (the _unseen file) are a second
test set, never trained on, by the probe or by the finetune baseline.

Controls, all reported beside the real number:
    control     Hewitt & Liang (2019) control task: each distinct pitch-class
                set of the step just heard gets a random label drawn from the
                task's own class distribution. Same probe, same lam.
                selectivity = real test acc - control test acc.
    position    the same probe on a one-hot of the bar's index in the window
                alone. Phase is position mod 4 when windows start on a I bar;
                this says how much of the score position alone buys.
    majority    the most frequent training class.
    layer 0     in every curve: the previous step alone, no context.
    random      the random-init model, if extracted.

MDL (Voita & Titov 2020), online code, on split seed 0: train on the first
0.1%, 0.2%, 0.4% ... of the shuffled training set and code the next block in
bits; compression = uniform code / online code.

    python -m midi_adapter.probe_train --probe_dir $WORK/probe_r1 \\
        --models base finetune random

Writes <probe_dir>/results.csv (one row per model, task, layer, seed) and
<probe_dir>/summary.csv (mean over seeds), and prints the best layer per model.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as F

TASKS = {'roman': 3, 'next_chord': 24}
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
MDL_FRACS = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1.0]


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def class_weights(y: torch.Tensor, K: int) -> torch.Tensor:
    n = torch.bincount(y, minlength=K).float()
    w = torch.where(n > 0, n.sum() / (K * n.clamp(min=1)), torch.zeros_like(n))
    return w


def fit(X: torch.Tensor, y: torch.Tensor, K: int, lam: float,
        weight: torch.Tensor | None = None, max_iter: int = 200):
    """(W, c) minimising weighted CE + lam * ||W||^2."""
    D = X.shape[1]
    W = torch.zeros(D, K, device=X.device, requires_grad=True)
    c = torch.zeros(K, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, c], lr=1.0, max_iter=max_iter,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + c, y, weight=weight) + lam * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), c.detach()


def logits(X, Wc):
    return X @ Wc[0] + Wc[1]


def acc_f1(z: torch.Tensor, y: torch.Tensor, K: int):
    if len(y) == 0:
        return float('nan'), float('nan')
    pred = z.argmax(1)
    acc = (pred == y).float().mean().item()
    f1s = []
    for k in range(K):
        tp = ((pred == k) & (y == k)).sum().item()
        fp = ((pred == k) & (y != k)).sum().item()
        fn = ((pred != k) & (y == k)).sum().item()
        if tp + fn == 0:            # class absent from this test set
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return acc, float(np.mean(f1s))


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-4)
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in others]


def choose_lambda(X, y, groups, K, folds: int, seed: int) -> float:
    """lam with the lowest mean validation CE over file-grouped folds."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % folds))
    f = np.array([fold_of[g] for g in groups])
    best, best_ce = LAMBDAS[0], float('inf')
    for lam in LAMBDAS:
        ces = []
        for k in range(folds):
            tr, va = np.where(f != k)[0], np.where(f == k)[0]
            if len(va) == 0:
                continue
            Xtr, Xva = standardise(X[tr], X[va])
            Wc = fit(Xtr, y[tr], K, lam, class_weights(y[tr], K))
            ces.append(F.cross_entropy(logits(Xva, Wc), y[va]).item())
        ce = float(np.mean(ces))
        if ce < best_ce:
            best, best_ce = lam, ce
    return best


def online_codelength(X, y, K, lam, seed: int):
    """(online code bits, uniform code bits) on X, y in a seeded order."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(y), generator=g).to(X.device)
    X, y = X[order], y[order]
    ends = sorted({max(int(round(fr * len(y))), K) for fr in MDL_FRACS})
    ends[-1] = len(y)
    bits = ends[0] * np.log2(K)                     # first block: uniform code
    for a, b in zip(ends[:-1], ends[1:]):
        Xtr, Xnext = standardise(X[:a], X[a:b])
        Wc = fit(Xtr, y[:a], K, lam, class_weights(y[:a], K))
        bits += F.cross_entropy(logits(Xnext, Wc), y[a:b], reduction='sum').item() / np.log(2)
    return bits, len(y) * np.log2(K)


# ---------------------------------------------------------------------------
# Labels and splits
# ---------------------------------------------------------------------------

def control_labels(pcset: np.ndarray, y: np.ndarray, K: int, train: np.ndarray,
                   seed: int = 1234) -> np.ndarray:
    """Hewitt & Liang control task: one random label per distinct input type
    (the pitch-class set just heard), drawn from the real class distribution."""
    rng = np.random.default_rng(seed + K)
    prior = np.bincount(y[train], minlength=K) / max(train.sum(), 1)
    types = np.unique(pcset)
    lab = dict(zip(types, rng.choice(K, size=len(types), p=prior)))
    return np.array([lab[t] for t in pcset])


def file_split(meta, seed: int, test_frac: float = 0.2):
    """(train, test, heldout) boolean masks. Split by source file, never by span."""
    seen = meta['split'] == 'finetune'
    files = np.unique(meta['src_file'][seen])
    rng = np.random.default_rng(seed)
    test_files = set(rng.permutation(files)[:max(1, int(round(test_frac * len(files))))])
    is_test = np.array([f in test_files for f in meta['src_file']])
    return seen & ~is_test, seen & is_test, meta['split'] == 'unseen'


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode())


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--probe_dir', required=True)
    p.add_argument('--models', nargs='+', default=['base', 'finetune', 'random'])
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--layers', default='all', help="'all' or e.g. 0,4,8,12")
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--cv_folds', type=int, default=5)
    p.add_argument('--no_mdl', action='store_true')
    p.add_argument('--no_control', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = dict(np.load(os.path.join(args.probe_dir, 'meta.npz')))
    for k in ('split', 'src_file'):
        meta[k] = meta[k].astype(str)
    groups = np.array([stable_hash(f) for f in meta['src_file']])
    n_seen = int((meta['split'] == 'finetune').sum())
    print(f'{len(groups)} examples: {n_seen} trained tonics, '
          f'{len(groups) - n_seen} held-out tonics, device {device}')

    rows = []
    splits = {s: file_split(meta, s) for s in args.seeds}

    # --- model-free baselines: majority class and position only -------------
    W1 = int(meta['window_bars']) - 1
    pos_X = torch.from_numpy(np.eye(W1, dtype=np.float32)[meta['bar'] - 1]).to(device)
    for task in args.tasks:
        K = TASKS[task]
        y_all = torch.from_numpy(meta[task].astype(np.int64)).to(device)
        for s, (tr, te, ho) in splits.items():
            tr_t, te_t, ho_t = (torch.from_numpy(m).to(device) for m in (tr, te, ho))
            maj = torch.bincount(y_all[tr_t], minlength=K).argmax()
            Wc = fit(pos_X[tr_t], y_all[tr_t], K, 1e-4, class_weights(y_all[tr_t], K))
            for name, z_te, z_ho in (
                    ('majority', F.one_hot(maj, K).float().expand(int(te_t.sum()), K),
                     F.one_hot(maj, K).float().expand(int(ho_t.sum()), K)),
                    ('position', logits(pos_X[te_t], Wc), logits(pos_X[ho_t], Wc))):
                a, f1 = acc_f1(z_te, y_all[te_t], K)
                ah, fh = acc_f1(z_ho, y_all[ho_t], K)
                rows.append(dict(model=name, task=task, layer='', seed=s,
                                 test_acc=a, test_f1=f1, heldout_acc=ah, heldout_f1=fh))
        print(f'  {task}: position-only test acc '
              f'{np.mean([r["test_acc"] for r in rows if r["model"] == "position" and r["task"] == task]):.3f}')

    # --- the probes ---------------------------------------------------------
    for model in args.models:
        path = os.path.join(args.probe_dir, model, 'feats.npy')
        if not os.path.exists(path):
            print(f'  {model}: no {path}, skipped')
            continue
        feats = np.load(path, mmap_mode='r')
        L1 = feats.shape[1]
        layers = range(L1) if args.layers == 'all' else [int(l) for l in args.layers.split(',')]
        for layer in layers:
            X = torch.from_numpy(np.ascontiguousarray(feats[:, layer]).astype(np.float32)).to(device)
            for task in args.tasks:
                K = TASKS[task]
                y_np = meta[task].astype(np.int64)
                y = torch.from_numpy(y_np).to(device)
                tr0 = splits[args.seeds[0]][0]
                idx0 = np.where(tr0)[0]
                lam = choose_lambda(X[idx0], y[idx0], groups[idx0], K,
                                    args.cv_folds, args.seeds[0])
                for s, (tr, te, ho) in splits.items():
                    Xtr, Xte, Xho = standardise(X[tr], X[te], X[ho])
                    w = class_weights(y[tr], K)
                    Wc = fit(Xtr, y[tr], K, lam, w)
                    a, f1 = acc_f1(logits(Xte, Wc), y[te], K)
                    ah, fh = acc_f1(logits(Xho, Wc), y[ho], K)
                    # test spans the finetune baseline never trained on
                    fresh = te & (meta['ft_split'] < 2)
                    Xfr = standardise(X[tr], X[fresh])[1]
                    af, _ = acc_f1(logits(Xfr, Wc), y[fresh], K)
                    r = dict(model=model, task=task, layer=layer, seed=s, lam=lam,
                             test_acc=a, test_f1=f1, heldout_acc=ah, heldout_f1=fh,
                             test_acc_unseen_by_ft=af)
                    if not args.no_control:
                        yc = torch.from_numpy(control_labels(
                            meta['prev_pcset'], y_np, K, tr)).to(device)
                        Wc_c = fit(Xtr, yc[tr], K, lam, class_weights(yc[tr], K))
                        ac, _ = acc_f1(logits(Xte, Wc_c), yc[te], K)
                        r.update(control_acc=ac, selectivity=a - ac)
                    if not args.no_mdl and s == args.seeds[0]:
                        bits, uni = online_codelength(X[tr], y[tr], K, lam, s)
                        r.update(mdl_bits=bits, uniform_bits=uni, compression=uni / bits)
                    rows.append(r)
                last = [r for r in rows if r['model'] == model and r['task'] == task
                        and r['layer'] == layer]
                print(f'  {model:>9} L{layer:<2} {task:<10} lam={lam:g}  '
                      f'test {np.mean([r["test_acc"] for r in last]):.3f}  '
                      f'held-out {np.mean([r["heldout_acc"] for r in last]):.3f}  '
                      f'sel {np.mean([r.get("selectivity", np.nan) for r in last]):.3f}',
                      flush=True)
            del X

    # --- write --------------------------------------------------------------
    keys = ['model', 'task', 'layer', 'seed', 'lam', 'test_acc', 'test_f1',
            'heldout_acc', 'heldout_f1', 'test_acc_unseen_by_ft', 'control_acc',
            'selectivity', 'mdl_bits', 'uniform_bits', 'compression']
    out = os.path.join(args.probe_dir, 'results.csv')
    with open(out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, '') for k in keys})
    print(f'\nwrote {out}')

    num = [k for k in keys[5:]]
    summary = {}
    for r in rows:
        summary.setdefault((r['model'], r['task'], r['layer']), []).append(r)
    out2 = os.path.join(args.probe_dir, 'summary.csv')
    with open(out2, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['model', 'task', 'layer', 'n_seeds'] + [f'{k}_mean' for k in num]
                    + ['test_acc_sd', 'heldout_acc_sd'])
        for (m, t, l), rs in summary.items():
            def mean(k):
                v = [float(r[k]) for r in rs if r.get(k, '') != '']
                return np.mean(v) if v else ''
            def sd(k):
                v = [float(r[k]) for r in rs if r.get(k, '') != '']
                return np.std(v) if len(v) > 1 else ''
            wr.writerow([m, t, l, len(rs)] + [mean(k) for k in num]
                        + [sd('test_acc'), sd('heldout_acc')])
    print(f'wrote {out2}')

    print('\nbest layer by test accuracy (mean over seeds)')
    print(f'{"model":<10}{"task":<12}{"layer":>6}{"test":>8}{"held-out":>10}{"select.":>9}')
    for model in args.models + ['position', 'majority']:
        for task in args.tasks:
            cand = [(k, rs) for k, rs in summary.items() if k[0] == model and k[1] == task]
            if not cand:
                continue
            (m, t, l), rs = max(cand, key=lambda kv: np.mean([r['test_acc'] for r in kv[1]]))
            sel = [r['selectivity'] for r in rs if 'selectivity' in r]
            print(f'{m:<10}{t:<12}{str(l):>6}{np.mean([r["test_acc"] for r in rs]):>8.3f}'
                  f'{np.mean([r["heldout_acc"] for r in rs]):>10.3f}'
                  f'{(np.mean(sel) if sel else float("nan")):>9.3f}')


if __name__ == '__main__':
    main()
