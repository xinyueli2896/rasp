"""
Stage-2-only orchestration wrapper for Structured-Arrangement.

Bypasses Stage 1 (phrase retrieval + PolyDisVAE piano regeneration) and feeds
our snippet's accompaniment tracks directly to the multi-track orchestrator.
Use this when your snippets are already a valid piano accompaniment and you
just want to fan them out into instrumental band arrangements.

Track detection (case-insensitive on `inst.name`):
  * Melody   — track 0 OR name containing "melody" / "mel" / "lead"
               → used as the melody overlay in the final MIDI; NOT sent to Stage 2
  * Everything else (piano, bridge, chord, unnamed, ...)
               → merged into acc_piano and fed to the orchestrator

Setup
-----
Requires Structured-Arrangement-Code cloned locally with data_file_dir/
checkpoints in place (same as batch_arrange.py). Run from INSIDE that repo:

  cd /path/to/Structured-Arrangement-Code && conda activate sarr
  python /path/to/rasp/midi_adapter/batch_orchestrate.py \\
      --in_dir    /path/to/pop909_ivvi_snippets \\
      --out_dir   /path/to/pop909_ivvi_orchestrated_s2only \\
      --data_root data_file_dir/ \\
      --n_bars 4 --tempo 120 --num_sample 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pretty_midi as pyd
from scipy.interpolate import interp1d

# Users run this from inside Structured-Arrangement-Code, so add cwd to
# sys.path (invoking a script by full path only adds the script's directory).
sys.path.insert(0, os.getcwd())

from arrangement_utils import load_premise, prompt_sampling, orchestration   # type: ignore[import-not-found]


MELODY_KEYWORDS = ('melody', 'mel', 'lead')
SUBBEATS_PER_BAR = 16   # 4 beats × 4 subbeats per beat (ACC=4 in their code)
SEG_BARS         = 2    # AccoMontage's orchestrator operates on 2-bar segments


def _is_melody_track(inst: pyd.Instrument, track_id: int) -> bool:
    name = (inst.name or '').lower().strip()
    return track_id == 0 or any(kw in name for kw in MELODY_KEYWORDS)


def _pm_to_acc_piano(pm: pyd.PrettyMIDI, n_bars: int) -> np.ndarray:
    """Convert a snippet MIDI into the acc_piano array Stage 2 expects.

    Returns numpy array of shape (n_segments, T_per_seg, 128) where
      n_segments = n_bars // SEG_BARS
      T_per_seg  = SEG_BARS * SUBBEATS_PER_BAR   (= 32 at 2 bars × 16 subbeats)
      128        = MIDI pitches

    Cell value = note duration in subbeats (matches AccoMontage's midi2matrix
    convention). Melody tracks are excluded; every other track is merged."""
    beats = pm.get_beats()
    if len(beats) < 2:
        raise ValueError('MIDI has fewer than 2 beats')
    # Extend by one beat interval so the very last subbeat still interpolates.
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))

    ACC   = SUBBEATS_PER_BAR // 4        # subbeats per beat = 4
    total_subbeats = n_bars * SUBBEATS_PER_BAR
    quantize = interp1d(np.arange(len(beats)) * ACC, beats, kind='linear',
                        fill_value='extrapolate')
    quaver = quantize(np.arange(total_subbeats))

    merged = np.zeros((total_subbeats, 128), dtype=np.float32)
    for i, inst in enumerate(pm.instruments):
        if _is_melody_track(inst, i):
            continue
        if inst.is_drum:
            continue
        for note in inst.notes:
            s = int(np.argmin(np.abs(quaver - note.start)))
            e = int(np.argmin(np.abs(quaver - note.end)))
            if e == s:
                e = min(s + 1, total_subbeats - 1)
            # Keep the longer of any duplicate onsets on the same (subbeat, pitch).
            merged[s, note.pitch] = max(merged[s, note.pitch], e - s)

    T_per_seg  = SEG_BARS * SUBBEATS_PER_BAR
    n_segments = total_subbeats // T_per_seg
    if n_segments < 2:
        raise ValueError(
            f'n_bars={n_bars} → {n_segments} segment(s); orchestrator needs ≥ 2')
    return merged[:n_segments * T_per_seg].reshape(n_segments, T_per_seg, 128)


def _extract_melody_notes(pm: pyd.PrettyMIDI, n_bars: int) -> list[pyd.Note]:
    """Return the melody track's notes, clipped to the first n_bars*SUBBEATS_PER_BAR
    subbeats. Uses the same subbeat grid as _pm_to_acc_piano so overlay lines up."""
    beats = pm.get_beats()
    if len(beats) < 2:
        return []
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    ACC   = SUBBEATS_PER_BAR // 4
    total_subbeats = n_bars * SUBBEATS_PER_BAR
    quantize = interp1d(np.arange(len(beats)) * ACC, beats, kind='linear',
                        fill_value='extrapolate')
    quaver = quantize(np.arange(total_subbeats + 1))
    t_end = float(quaver[-1])

    for i, inst in enumerate(pm.instruments):
        if _is_melody_track(inst, i):
            return [pyd.Note(velocity=n.velocity, pitch=n.pitch,
                             start=n.start, end=min(n.end, t_end))
                    for n in inst.notes if n.start < t_end]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',    required=True,
                   help='Folder of snippet MIDIs (from `filter_nottingham leadsheets`)')
    p.add_argument('--out_dir',   required=True)
    p.add_argument('--data_root', required=True,
                   help='Structured-Arrangement data_file_dir/ with checkpoints')
    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--n_bars',     type=int, default=4,
                   help='Bars per snippet. Must be an even multiple of 2 and ≥ 4.')
    p.add_argument('--tempo',      type=int, default=120)
    p.add_argument('--num_sample', type=int, default=2,
                   help='Band arrangements to produce per snippet')
    p.add_argument('--must_have',    type=int, nargs='*', default=[0, 24],
                   help='Programs the orchestrator must include (Slakh IDs)')
    p.add_argument('--mustnot_have', type=int, nargs='*', default=[64])
    p.add_argument('--no_prompt', action='store_true',
                   help='Disable 2-bar orchestral style prompt')
    p.add_argument('--limit',      type=int, default=0)
    args = p.parse_args()

    if args.n_bars % 2 != 0 or args.n_bars < 4:
        raise SystemExit(f'n_bars must be even and ≥ 4 (got {args.n_bars})')

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading Stage 2 (orchestrator only, skipping piano arranger) ...')
    _, orchestrator, _, band_prompt = load_premise(
        args.data_root, args.device, load_piano_arranger=False)

    files = sorted(f for f in os.listdir(args.in_dir)
                   if f.lower().endswith(('.mid', '.midi')))
    if args.limit:
        files = files[:args.limit]
    print(f'Found {len(files)} snippet MIDIs in {args.in_dir}')

    n_ok = 0
    for i, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        src  = os.path.join(args.in_dir, fname)
        song_dir = os.path.join(args.out_dir, stem)
        os.makedirs(song_dir, exist_ok=True)
        shutil.copy(src, os.path.join(song_dir, 'lead sheet.mid'))

        try:
            pm         = pyd.PrettyMIDI(src)
            acc_piano  = _pm_to_acc_piano(pm, n_bars=args.n_bars)
            mel_notes  = _extract_melody_notes(pm, n_bars=args.n_bars)
        except Exception as e:
            print(f'  SKIP {fname}: acc_piano build failed — {e}')
            continue

        try:
            func_prompt = prompt_sampling(
                acc_piano, *band_prompt, args.must_have, args.mustnot_have, args.device)
            if args.no_prompt:
                instruments, _ = func_prompt
                time_prompt    = None
            else:
                instruments, time_prompt = func_prompt

            midi_collection = orchestration(
                acc_piano, None, instruments, time_prompt, orchestrator, args.device,
                blur=0.25, p=0.05, t=6, tempo=args.tempo, num_sample=args.num_sample)

            mel_track = pyd.Instrument(program=82, is_drum=False, name='melody')
            mel_track.notes = mel_notes
            for idx, piece in enumerate(midi_collection):
                piece.instruments = [mel_track] + piece.instruments
                piece.write(os.path.join(song_dir, f'arrangement_band-{idx:02d}.mid'))
        except Exception as e:
            print(f'  SKIP {fname}: Stage 2 failed — {e}')
            continue

        n_ok += 1
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(files)} processed ({n_ok} succeeded)')

    print(f'\nDone. {n_ok}/{len(files)} succeeded.')
    print(f'Outputs: {args.out_dir}/<stem>/arrangement_band-NN.mid')


if __name__ == '__main__':
    main()
