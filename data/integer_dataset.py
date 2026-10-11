"""Data config for the integer experiment — CHORD-MATCHED split structure.

    token[i] = (x + OFFSETS[i % 4]) % 12,   x = token[0],   OFFSETS = [0,5,7,0]

THE SPLIT, AND WHY IT HAS ONLY TWO CELLS
----------------------------------------
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

# Chord-matched: one pool, both stages. Not a typo -- see the module docstring.
PRETRAIN_STARTERS = SEEN
ADAPTER_STARTERS  = SEEN

# 4 cycles * 8 = 32 tokens. Teacher forcing drops one, so the AR sees 31 and
# the rule model is fed 31 + BOS = 32, which is exactly the length its
# compiled `indices` one-hot supports (models/compiled_rules/..._T32_*.pt).
# Changing N_CYCLES means recompiling the rule model.
N_CYCLES = 8
SEQ_LEN  = PERIOD * N_CYCLES


def validate_split(seen=None, unseen=None, n_cycles: int = N_CYCLES) -> dict:
    """Check the holdout is structural, not lexical. Raises if it is not."""
    seen   = list(SEEN if seen is None else seen)
    unseen = list(UNSEEN if unseen is None else unseen)
    T      = PERIOD * n_cycles

    overlap = set(seen) & set(unseen)
    if overlap:
        raise ValueError(f'seen and unseen overlap: {sorted(overlap)}')

    cov = set()
    for x in seen:
        cov |= set(sequence([x], T)[0].tolist())

    novel = {u: sorted(set(sequence([u], T)[0].tolist()) - cov) for u in unseen}
    bad   = {u: v for u, v in novel.items() if v}
    if bad:
        raise ValueError(
            f'holdout is not purely structural: starter(s) {sorted(bad)} must '
            f'emit token(s) never produced by any seen starter: {bad}. Pick a '
            f'different holdout, or the unseen number conflates "cannot apply '
            f'the rule" with "has never emitted this symbol".')

    return dict(seen=seen, unseen=unseen, seq_len=T,
                token_coverage=sorted(cov),
                n_eval_sequences=len(unseen),
                n_eval_predictions=len(unseen) * (T - 1))


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
                    seen=None, unseen=None):
    """-> (train_loader, eval_loaders) where eval_loaders = {'seen', 'unseen'}.

    stage is 'pretrain' or 'adapter'; under this config they are the same
    pool, and that is deliberate.
    """
    if stage not in ('pretrain', 'adapter'):
        raise ValueError(f"stage must be 'pretrain' or 'adapter', got {stage!r}")
    info = validate_split(seen, unseen, n_cycles)

    train = IntegerRuleDataset(info['seen'], n_cycles, n_repeats)
    evals = {
        'seen':   IntegerRuleDataset(info['seen'],   n_cycles, 1),
        'unseen': IntegerRuleDataset(info['unseen'], n_cycles, 1),
    }
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        {k: DataLoader(v, batch_size=max(len(v), 1)) for k, v in evals.items()},
    )


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--unseen', type=int, nargs='+', default=UNSEEN)
    p.add_argument('--n_cycles', type=int, default=N_CYCLES)
    a = p.parse_args()

    seen = [x for x in range(VOCAB_SIZE) if x not in a.unseen]
    info = validate_split(seen, a.unseen, a.n_cycles)

    print(f'rule        : token[i] = (x + {OFFSETS}[i % {PERIOD}]) % {VOCAB_SIZE}')
    print(f'seq_len     : {info["seq_len"]}  ({a.n_cycles} cycles); AR sees '
          f'{info["seq_len"] - 1} after teacher forcing')
    print(f'\nsplit (chord-matched: ONE pool, both stages)')
    print(f'  AR pretrain : {info["seen"]}')
    print(f'  adapter     : {info["seen"]}   <- same pool, no pretrain-only cell')
    print(f'  UNSEEN      : {info["unseen"]}')
    print(f'\nevaluation partition')
    print(f'  seen    {len(info["seen"]):2d} starters')
    print(f'  unseen  {len(info["unseen"]):2d} starters = '
          f'{info["n_eval_predictions"]} token predictions  <- the whole result')
    print(f'\nholdout is structural, not lexical:')
    print(f'  tokens covered by seen starters: {info["token_coverage"]}')
    for u in info['unseen']:
        need = sorted(set(sequence([u], info['seq_len'])[0].tolist()))
        print(f'    unseen {u:2d} needs {need} -- all covered')
    if len(info['unseen']) < 4:
        print(f'\n  NOTE: {info["n_eval_predictions"]} predictions is a thin '
              f'result to rest a paper on. Widen with e.g.\n'
              f'  --unseen 2 6 8 10  (one of 34 valid 4-starter holdouts '
              f'containing 6 and 8),\n  or report over >= 5 seeds. Always run '
              f'this validator on a new holdout:\n  many plausible-looking '
              f'ones (e.g. 1 6 8 11) are not structural.')


if __name__ == '__main__':
    _main()
