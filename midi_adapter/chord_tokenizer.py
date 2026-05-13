"""
Chord tokenizer for bar-level rule model input.

Converts XF-MIDI chord annotations (from xf_midi.py) to integer token
sequences at bar granularity.

Vocabulary
----------
  token = root_semitone * N_QUALITIES + quality_idx   (0 .. N_ROOTS*N_QUALITIES - 1)
  NO_CHORD_TOKEN = N_ROOTS * N_QUALITIES              (last token)

  N_ROOTS     = 12  (chromatic semitones)
  N_QUALITIES = 35  (from CHORD_MAP in xf_midi.py)
  N_CHORD_TOKENS = 12 * 35 + 1 = 421
"""
from __future__ import annotations

import torch

# ---- quality vocabulary (must match xf_midi.CHORD_MAP order) ----
CHORD_MAP = [
    'maj', 'maj6', 'maj7', 'maj7(#11)', 'maj(9)', 'maj7(9)', 'maj6(9)', 'aug',
    'min', 'min6', 'min7', 'hdim7', 'min(9)', 'min7(9)', 'min7(11)', 'minmaj7',
    'minmaj7(9)', 'dim', 'dim7', '7', 'sus4(b7)', '(3,b5,b7)', '7(9)', '7(#11)',
    '7(13)', '7(b9)', '7(b13)', '7(#9)', 'aug(7)', 'aug(b7)', '1', '5', 'sus4',
    'sus2', '?',
]

N_ROOTS      = 12
N_QUALITIES  = len(CHORD_MAP)           # 35
N_CHORD_TOKENS = N_ROOTS * N_QUALITIES + 1
NO_CHORD_TOKEN = N_CHORD_TOKENS - 1    # 420  — also used as padding

# normalise any enharmonic spelling to semitone index
_NOTE_TO_SEMI = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0,
}
_QUALITY_TO_IDX = {q: i for i, q in enumerate(CHORD_MAP)}


def chord_str_to_token(chord_str: str | None) -> int:
    """
    Convert an XF-MIDI chord string to an integer token.

    Examples
    --------
      'C:maj7'        -> root=0, quality=2  -> 0*35+2  = 2
      'F#:min'        -> root=6, quality=8  -> 6*35+8  = 218
      'N'             -> NO_CHORD_TOKEN (420)
      'D:7(9)/3'      -> strip inversion -> 'D:7(9)' -> 2*35+22 = 92
    """
    if not chord_str or chord_str == 'N':
        return NO_CHORD_TOKEN
    # strip inversion (e.g. '/3', '/5')
    chord_str = chord_str.split('/')[0]
    if ':' not in chord_str:
        return NO_CHORD_TOKEN
    root_str, quality_str = chord_str.split(':', 1)
    semi = _NOTE_TO_SEMI.get(root_str)
    qidx = _QUALITY_TO_IDX.get(quality_str)
    if semi is None or qidx is None:
        return NO_CHORD_TOKEN
    return semi * N_QUALITIES + qidx


def chords_to_bar_tokens(
    xf_chords: list[list],      # [[subbeat_time_float, chord_str], ...]
    n_subbeats: int,
    subbeats_per_bar: int = 16,
) -> list[int]:
    """
    Snap XF-MIDI chord events to bar-level token sequence.

    Parameters
    ----------
    xf_chords        : midi.chords from XFMidi (time is in subbeats when
                       XFMidi was constructed with constant_tempo=60/beat_div)
    n_subbeats       : total length of the MIDI sequence in subbeats
    subbeats_per_bar : 16 for 4/4 at beat_div=4

    Returns
    -------
    List of length ceil(n_subbeats / subbeats_per_bar), one token per bar.
    """
    n_bars = (n_subbeats + subbeats_per_bar - 1) // subbeats_per_bar
    tokens = [NO_CHORD_TOKEN] * n_bars

    for time, chord_str in xf_chords:
        b = int(time) // subbeats_per_bar
        if 0 <= b < n_bars:
            tokens[b] = chord_str_to_token(chord_str)

    # forward-fill: if a bar has no annotation, inherit the previous chord
    last = NO_CHORD_TOKEN
    for b in range(n_bars):
        if tokens[b] != NO_CHORD_TOKEN:
            last = tokens[b]
        elif last != NO_CHORD_TOKEN:
            tokens[b] = last

    return tokens


def bar_tokens_tensor(
    xf_chords: list[list],
    n_subbeats: int,
    subbeats_per_bar: int = 16,
    device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """Convenience wrapper — returns a 1-D int64 tensor of bar chord tokens."""
    tokens = chords_to_bar_tokens(xf_chords, n_subbeats, subbeats_per_bar)
    return torch.tensor(tokens, dtype=torch.long, device=device)


def chords_to_beat_tokens(
    xf_chords: list[list],
    n_subbeats: int,
) -> list[int]:
    """
    One chord token per subbeat (beat-level granularity).

    With BEAT_DIV=1 every subbeat is one quarter note, so each beat gets its
    own chord token.  This is correct for our synthetic data where the chord
    changes on every beat (I-IV-V-I within each bar).

    Equivalent to chords_to_bar_tokens(..., subbeats_per_bar=1).
    """
    return chords_to_bar_tokens(xf_chords, n_subbeats, subbeats_per_bar=1)
