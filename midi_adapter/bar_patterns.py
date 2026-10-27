"""The five bar realisations, and how to recognise one.

Shared so that the data checker and the evaluation sweeps classify bars the
same way. Duplicating this is how `walking` came to be matched only in its
major form, which read as "the model never plays walking" for a while.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.generate_synthetic_bass import COMMON_CHORDS

# Semitones above the bar's root; None is a beat with no onset. Each name maps
# to every shape that realisation can take.
#
# Only `walking` has more than one: _bar_notes walks it as
# tones[i % len(tones)], and `tones` is COMMON_CHORDS[quality] for a quality
# drawn at random per song, so a minor bar is 0 3 7 0 and a seventh 0 4 7 10.
# The other four use the root and tones[2], which is 7 in every quality.
_WALK = [[t[i % len(t)] for i in range(4)] for t in COMMON_CHORDS.values()]

PATTERNS: dict[str, list[list[int | None]]] = {
    'root4':   [[0, 0, 0, 0]],
    'root2':   [[0, None, 0, None]],
    'whole':   [[0, None, None, None]],
    'half':    [[0, None, 7, None]],
    'walking': _WALK,
}


def classify_bar(pcs) -> str | None:
    """Which realisation this bar's shape is, or None.

    Measured against the bar's OWN downbeat, not an expected root, so a
    correctly-shaped bar on the wrong chord still classifies. A bar whose
    downbeat has no note cannot be referenced and is unclassified.
    """
    if not pcs or pcs[0] is None:
        return None
    r = pcs[0]
    steps = [None if p is None else (p - r) % 12 for p in pcs]
    for name, shapes in PATTERNS.items():
        if steps in shapes:
            return name
    return None
