"""Data config for the integer experiment — CHORD-MATCHED split structure.

    token[i] = (x + OFFSETS[i % 4]) % 12,   x = token[0],   OFFSETS = [0,5,7,0]

THE SPLIT
---------
The old config (`data/dataset.py`) ran two training stages over DIFFERENT
starter sets -- AR pretrain on {0,5,9}, adapter on {0,1,2,3,4,7,10,11} -- which
gives a four-way evaluation partition: pretrain-only, finetune-only, both,
neither. That partition can measure catastrophic forgetting, because there are
starters the AR learned and the adapter never touched.

The chord experiment cannot measure forgetting: its base is pretrained on Lakh
(general music) and no key is rule-following, so "pretrain-only" is empty and
the partition collapses to seen / unseen. To make the two experiments
comparable, this config matches that shape:

    AR pretrain  : SEEN   (all 10)
    adapter      : SEEN   (all 10)      <- same pool, so A\\B and B\\A are empty
    evaluation   : SEEN vs UNSEEN

Here the AR is pretrained on 100% rule-following data (unlike chord, whose
base never sees the rule), so the question is not "can the adapter install a
rule from nothing" but "the AR has learned the rule on 10 starters and still
fails on 2 -- can a rule-guided adapter close that gap". That is the question
the chord unseen-key number actually answers too.

WHAT "UNSEEN" MEANS
-------------------
Not an unseen SYMBOL -- an unseen BINDING. Every token an unseen starter has
to emit already appears inside the seen starters' sequences:

    seen starters cover tokens [0..11]   (all of them)
    unseen 6 needs [1, 6, 11]   all seen
    unseen 8 needs [1, 3, 8]    all seen

So the difficulty is structural, not vocabulary coverage. `validate_split`
enforces this, and refuses a split that smuggles in a novel token --
`report.tex` chose its V=24 holdout {17,19,20,21} specifically to exclude
token 23 for this reason, and that care is worth keeping.

THE COST OF THIS SHAPE, STATED UP FRONT
---------------------------------------
Sequences are DETERMINISTIC given the starter, so the evaluation set is
exactly as large as the holdout:

    UNSEEN = {6, 8}  ->  2 sequences  ->  62 token predictions, total.

There is no second axis to resample over (the chord experiment has many
windows per key, and even there n=28 was too thin to resolve 0.565 against a
0.500 bar). Error bars can only come from training seeds.

Two ways to live with that, both supported:

  * keep UNSEEN = {6, 8} for an exact match to the chord holdout, and report
    over >= 5 seeds. This is what `report.tex` did.
  * widen the holdout. Exactly 34 four-starter holdouts contain {6,8} and
    still pass `validate_split`, e.g. UNSEEN = [2, 6, 8, 10]. That doubles
    the eval set for two training starters, and {6,8} stays reportable as
    the chord-comparable subset.

    Check any candidate -- do not eyeball it. [1, 6, 8, 11] looks fine and
    is NOT valid: removing 1 and 11 takes tokens 1 and 6 out of the seen
    set, so three of the four held-out starters would have to emit a symbol
    never produced in training, and the unseen number would no longer be
    measuring rule application.

`python -m data.integer_dataset` prints the config and runs the validator.
"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasp_program.next_token_rule import (  # noqa: E402
    OFFSETS, PERIOD, VOCAB_SIZE, next_token, sequence,
)

# Held out of BOTH stages. Matches the chord experiment's unseen keys exactly
# (filter_nottingham.py: UNSEEN_KEYS_DEFAULT = (6, 8)).
UNSEEN   = [6, 8]
SEEN     = [x for x in range(VOCAB_SIZE) if x not in UNSEEN]

# The two training stages draw from DIFFERENT pools, so all four evaluation
# cells are populated:
#
#   A \ B  {5, 9, 11}        pretrain-only  -> catastrophic forgetting
#   B \ A  {1, 3, 4, 7, 10}  finetune-only  -> rule installation
#   A n B  {0, 2}            both           -> control, should stay ~1.0
#   none   {6, 8}            unseen         -> generalisation (the headline)
#
# Why A is small (5 of 12). The rule is translation-equivariant in x, so
# every starter is structurally identical and only the COUNT matters. Give
# the AR too many and it learns the rule outright, generalises to all twelve
# by itself, and there is nothing left for the adapter to demonstrate. The
# V=24 report used 6 of 24 and its pretrain baseline still scored 0.000 on
# the unseen cell; 5 of 12 is a far higher fraction, so this is the number
# to check empirically before trusting any adapter result. See
# `ar_baseline_is_informative` below.
PRETRAIN_STARTERS = [0, 2, 5, 9, 11]                 # A
ADAPTER_STARTERS  = [0, 1, 2, 3, 4, 7, 10]           # B

# 4 cycles * 8 = 32 tokens. Teacher forcing drops one, so the AR sees 31 and
# the rule model is fed 31 + BOS = 32, which is exactly the length its
# compiled `indices` one-hot supports (models/compiled_rules/..._T32_*.pt).
# Changing N_CYCLES means recompiling the rule model.
N_CYCLES = 8
SEQ_LEN  = PERIOD * N_CYCLES


def validate_split(pretrain=None, adapter=None, unseen=None,
                   n_cycles: int = N_CYCLES) -> dict:
    """Check the four-way split. Raises if it is malformed or leaky."""
    A      = list(PRETRAIN_STARTERS if pretrain is None else pretrain)
    B      = list(ADAPTER_STARTERS  if adapter  is None else adapter)
    unseen = list(UNSEEN if unseen is None else unseen)
    T      = PERIOD * n_cycles
    seen   = sorted(set(A) | set(B))

    leak = set(unseen) & set(seen)
    if leak:
        raise ValueError(
            f'starter(s) {sorted(leak)} are held out AND trained on. The '
            f'unseen number would be meaningless.')

    cells = dict(pretrain_only=sorted(set(A) - set(B)),
                 finetune_only=sorted(set(B) - set(A)),
                 both=sorted(set(A) & set(B)),
                 unseen=sorted(unseen))
    empty = [k for k, v in cells.items() if not v]
    if empty:
        raise ValueError(
            f'evaluation cell(s) {empty} are empty. pretrain_only measures '
            f'forgetting and finetune_only measures installation, so a split '
            f'without them answers a different question than the four-way '
            f'partition claims to.')

    cov = set()
    for x in seen:
        cov |= set(sequence([x], T)[0].tolist())

    novel = {u: sorted(set(sequence([u], T)[0].tolist()) - cov) for u in unseen}
    bad   = {u: v for u, v in novel.items() if v}
    if bad:
        raise ValueError(
            f'holdout is not purely structural: starter(s) {sorted(bad)} must '
            f'emit token(s) never produced by any TRAINED starter: {bad}. Pick '
            f'a different holdout, or the unseen number conflates "cannot '
            f'apply the rule" with "has never emitted this symbol".')

    return dict(pretrain=A, adapter=B, cells=cells,
                seen=seen, unseen=unseen, seq_len=T,
                token_coverage=sorted(cov),
                n_predictions={k: len(v) * (T - 1) for k, v in cells.items()})


class IntegerRuleDataset(Dataset):
    """Teacher-forced next-token pairs: input = seq[:-1], target = seq[1:].

    `n_repeats` only duplicates rows so a batch can be assembled; the
    sequences are deterministic, so it adds no information whatsoever. It is
    named `n_repeats` rather than `n_seqs_per_starter` to stop that reading
    like data augmentation.
    """

    def __init__(self, starters, n_cycles: int = N_CYCLES, n_repeats: int = 1):
        self.starters = list(starters)
        seqs = torch.as_tensor(sequence(self.starters, PERIOD * n_cycles))
        if n_repeats > 1:
            seqs = seqs.repeat(n_repeats, 1)
        self.inputs  = seqs[:, :-1].contiguous().long()
        self.targets = seqs[:, 1:].contiguous().long()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


def get_dataloaders(stage: str = 'adapter', batch_size: int = 64,
                    n_cycles: int = N_CYCLES, n_repeats: int = 200,
                    pretrain=None, adapter=None, unseen=None):
    """-> (train_loader, eval_loaders) with one eval loader PER CELL.

    stage picks which pool to train on: 'pretrain' trains on A, 'adapter'
    on B. The eval loaders are the same four cells either way, so the AR
    baseline and the adapter are scored on identical partitions.
    """
    if stage not in ('pretrain', 'adapter'):
        raise ValueError(f"stage must be 'pretrain' or 'adapter', got {stage!r}")
    info = validate_split(pretrain, adapter, unseen, n_cycles)

    pool  = info['pretrain'] if stage == 'pretrain' else info['adapter']
    train = IntegerRuleDataset(pool, n_cycles, n_repeats)
    evals = {k: IntegerRuleDataset(v, n_cycles, 1)
             for k, v in info['cells'].items()}
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        {k: DataLoader(v, batch_size=max(len(v), 1)) for k, v in evals.items()},
    )


def ar_baseline_is_informative(acc_by_cell: dict, threshold: float = 0.5) -> str:
    """Read the AR-alone numbers before trusting any adapter result.

    The experiment only has something to show if the AR, trained on A,
    FAILS outside it. Pass the AR baseline's per-cell accuracy here and this
    says whether the split is doing its job.
    """
    ft = acc_by_cell.get('finetune_only', 0.0)
    un = acc_by_cell.get('unseen', 0.0)
    if un > threshold and ft > threshold:
        return (f'AR alone already scores {un:.3f} unseen / {ft:.3f} '
                f'finetune-only: it learned the RULE, not the starters, so '
                f'there is no headroom for the adapter. SHRINK the pretrain '
                f'pool (PRETRAIN_STARTERS) and retrain.')
    if un > threshold:
        return (f'AR alone scores {un:.3f} on unseen. The headline cell is '
                f'already solved before the adapter runs; shrink A.')
    return (f'AR alone: {un:.3f} unseen, {ft:.3f} finetune-only -- both low, '
            f'so both cells have headroom. Split is doing its job.')


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pretrain', type=int, nargs='+', default=PRETRAIN_STARTERS)
    p.add_argument('--adapter',  type=int, nargs='+', default=ADAPTER_STARTERS)
    p.add_argument('--unseen',   type=int, nargs='+', default=UNSEEN)
    p.add_argument('--n_cycles', type=int, default=N_CYCLES)
    a = p.parse_args()

    info = validate_split(a.pretrain, a.adapter, a.unseen, a.n_cycles)
    c, T = info['cells'], info['seq_len']

    print(f'rule        : token[i] = (x + {OFFSETS}[i % {PERIOD}]) % {VOCAB_SIZE}')
    print(f'seq_len     : {T}  ({a.n_cycles} cycles); AR sees {T - 1} after '
          f'teacher forcing')
    print(f'\ntraining pools')
    print(f'  A  AR pretrain : {info["pretrain"]}')
    print(f'  B  adapter     : {info["adapter"]}')

    print(f'\nevaluation partition')
    print(f'  {"":<9}{"cell":<16}{"starters":<26}{"n":>3}{"preds":>8}  measures')
    print('  ' + '-' * 83)
    what = {'pretrain_only': 'catastrophic forgetting',
            'finetune_only': 'rule installation',
            'both':          'control, should stay ~1.0',
            'unseen':        'GENERALISATION (the headline)'}
    for k in ('pretrain_only', 'finetune_only', 'both', 'unseen'):
        label = {'pretrain_only': r'A \ B', 'finetune_only': r'B \ A',
                 'both': 'A n B', 'unseen': 'neither'}[k]
        print(f'  {label:<9}{k:<16}{str(c[k]):<26}{len(c[k]):>3}'
              f'{info["n_predictions"][k]:>8}  {what[k]}')
    tot = sum(len(v) for v in c.values())
    print(f'  {"":<9}{"":<16}{"":<26}{tot:>3}  (of {VOCAB_SIZE} starters)')

    print(f'\nholdout is structural, not lexical:')
    print(f'  tokens covered by trained starters: {info["token_coverage"]}')
    for u in info['unseen']:
        need = sorted(set(sequence([u], T)[0].tolist()))
        print(f'    unseen {u:2d} needs {need} -- all covered')

    print(f'\nbefore trusting any adapter number, check the AR baseline:')
    print(f'  the AR trains on A ({len(info["pretrain"])} of {VOCAB_SIZE} '
          f'starters). The rule is translation-equivariant in x, so if that')
    print(f'  is enough for it to learn the RULE rather than the starters, it')
    print(f'  generalises everywhere and the adapter has nothing to add.')
    print(f'  Score the AR alone per cell and pass it to')
    print(f'  ar_baseline_is_informative(); shrink --pretrain if it says so.')
    if len(info['unseen']) < 4:
        print(f'\n  NOTE: unseen is {info["n_predictions"]["unseen"]} predictions. '
              f'Widen with --unseen 2 6 8 10, or report over >= 5 seeds.')


if __name__ == '__main__':
    _main()
