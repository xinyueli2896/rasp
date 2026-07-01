"""
Batch driver: feed each lead-sheet MIDI in --in_dir into Structured-Arrangement
and save the multi-track band arrangement(s).

Setup
-----
1. Clone the upstream repo and download their checkpoints:

     git clone https://github.com/zhaojw1998/Structured-Arrangement-Code
     cd Structured-Arrangement-Code
     # follow their README to create the conda env and `pip install -r requirements.txt`
     # download checkpoints zip from
     #   https://drive.google.com/file/d/1mk24C2uKcjmQ-jZQ0CxiFQm0lm3czSwC/view
     # decompress into ./data_file_dir/

2. Run this script from inside their repo so `arrangement_utils` is importable:

     cd /path/to/Structured-Arrangement-Code
     python /path/to/rasp/midi_adapter/batch_arrange.py \\
         --in_dir    /path/to/nottingham_ivvi_leadsheets/ \\
         --out_dir   /path/to/orchestrated/ \\
         --data_root data_file_dir/ \\
         --n_bars 8 --tempo 120 --num_sample 2

Inputs:  2-track lead-sheet MIDIs (track 0 = monophonic melody,
         track 1 = chord block) produced by
         `python -m midi_adapter.filter_nottingham leadsheets ...`

Output layout:  out_dir/<stem>/lead sheet.mid             (copied input)
                out_dir/<stem>/arrangement_piano.mid      (Stage 1 result)
                out_dir/<stem>/arrangement_band-{00..NN-1}.mid   (Stage 2 results)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pretty_midi as pyd

# When invoked by full path (`python /path/to/batch_arrange.py`), Python only
# adds the SCRIPT's directory to sys.path — not the cwd. Since the user is
# expected to run this from INSIDE the Structured-Arrangement-Code repo, we
# add cwd so `arrangement_utils` (and its subpackages) are importable.
sys.path.insert(0, os.getcwd())

# These imports require the Structured-Arrangement-Code repo on PYTHONPATH.
from arrangement_utils import (   # type: ignore[import-not-found]
    load_premise, read_lead_sheet, piano_arrangement,
    prompt_sampling, orchestration,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',    required=True,
                   help='Folder of 2-track lead-sheet MIDIs')
    p.add_argument('--out_dir',   required=True,
                   help='Where to write each song-folder of arrangements')
    p.add_argument('--data_root', required=True,
                   help='Folder containing the Structured-Arrangement checkpoint zip')
    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--n_bars',     type=int, default=8,
                   help='Bars per input lead sheet (used to build SEGMENTATION="A{n_bars}")')
    p.add_argument('--tempo',      type=int, default=120)
    p.add_argument('--num_sample', type=int, default=2,
                   help='Number of band arrangements per song')
    p.add_argument('--must_have',     type=int, nargs='*', default=[0, 24],
                   help='Programs the orchestrator must include (e.g. 0=Piano, 24=Guitar)')
    p.add_argument('--mustnot_have',  type=int, nargs='*', default=[64],
                   help='Programs the orchestrator must avoid (e.g. 64=Sax)')
    p.add_argument('--no_prompt', action='store_true',
                   help='Disable the 2-bar orchestral prompt (default uses one)')
    p.add_argument('--limit',     type=int, default=0,
                   help='Cap total songs processed (0 = no cap)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading Structured-Arrangement (Stage 1 piano_arranger + Stage 2 orchestrator) ...')
    piano_arranger, orchestrator, piano_texture, band_prompt = load_premise(
        args.data_root, args.device)

    files = sorted([f for f in os.listdir(args.in_dir)
                    if f.lower().endswith('.mid') or f.lower().endswith('.midi')])
    if args.limit:
        files = files[:args.limit]
    print(f'Found {len(files)} lead-sheet snippets in {args.in_dir}')

    # Single phrase covering the whole window (their format: <letter><bar_count>).
    segmentation = f'A{args.n_bars}'

    n_ok = 0
    for i, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        src  = os.path.join(args.in_dir, fname)
        # read_lead_sheet wants a {DEMO_ROOT}/{SONG_NAME}/lead sheet.mid layout.
        song_dir = os.path.join(args.out_dir, stem)
        os.makedirs(song_dir, exist_ok=True)
        shutil.copy(src, os.path.join(song_dir, 'lead sheet.mid'))

        try:
            lead_sheet = read_lead_sheet(args.out_dir, stem, segmentation, NOTE_SHIFT=0)
        except Exception as e:
            print(f'  SKIP {fname}: read_lead_sheet failed — {e}')
            continue

        # Stage 1: piano arrangement from the lead sheet
        rhythm_density = int(np.random.randint(2, 5))
        voice_number   = int(np.random.randint(2, 5))
        prefilter      = (rhythm_density, voice_number)

        try:
            midi_piano, acc_piano = piano_arrangement(
                *lead_sheet, *piano_texture, piano_arranger, prefilter, args.tempo)
            midi_piano.write(os.path.join(song_dir, 'arrangement_piano.mid'))
        except Exception as e:
            print(f'  SKIP {fname}: Stage 1 failed — {e}')
            continue

        # Stage 2: orchestrate the piano arrangement into a multi-track band
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
            mel_track.notes = midi_piano.instruments[0].notes
            for idx, piece in enumerate(midi_collection):
                piece.instruments = [mel_track] + piece.instruments
                piece.write(os.path.join(song_dir, f'arrangement_band-{idx:02d}.mid'))
        except Exception as e:
            print(f'  SKIP {fname}: Stage 2 failed — {e}')
            continue

        n_ok += 1
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(files)} processed  ({n_ok} succeeded)')

    print(f'\nDone. {n_ok}/{len(files)} succeeded.')
    print(f'Outputs: {args.out_dir}/<stem>/arrangement_band-NN.mid')


if __name__ == '__main__':
    main()
