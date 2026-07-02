"""
Convert batch_orchestrate output into a CP tensor training dataset.

Walks the AccoMontage output directory (one subfolder per snippet, each
containing `arrangement_band-NN.mid` files) and:

  1. Reads each band arrangement as a multi-instrument MIDI
  2. Aligns / quantizes it to 16th-note subbeats (AccoMontage renders at
     120 BPM constant, so we skip downbeat alignment — the first beat is
     already the bar start)
  3. Truncates to n_bars * SUBBEATS_PER_BAR subbeats (extra bars AccoMontage
     may append at the end are dropped)
  4. Transposes each window into every target_key using the smallest signed
     semitone shift (default = 10 seen keys, excluding F# and G# reserved
     for unseen-key evaluation)
  5. Emits the same 5 dataset files the trainer already knows how to load:
       {out_pt}.pt                     concatenated CP tensor
       {out_pt}.length.pt              per-window length
       {out_pt}.pitch_shift_range.pt   [0, 0] each (no runtime augmentation)
       {out_pt}.beat_chords.pt         subbeat-level chord tokens
       {out_pt}.txt                    index -> source filename mapping

Original key is parsed from the subfolder name (which is stamped by
`filter_nottingham leadsheets` in the pattern `..._key{X}_phase{P}_...`).

Song-level vs key-level eval — both axes are independent:
  * --target_keys        which pitch classes to transpose into
                         (default 10 seen keys, use `6 8` for unseen keys)
  * --val_song_frac      hold out entire source songs for out-of-song eval
                         (same seed → same partition; run with --split train,
                         then --split val to get aligned datasets)

Usage — training + 2×2 eval grid
-------------------------------
Training (train songs × 10 seen keys):
    python -m midi_adapter.extract_orchestrated \\
        --in_dir /path/to/pop909_ivvi_orchestrated_s2only \\
        --out_pt /path/to/data/pop909_orch_train \\
        --val_song_frac 0.1 --split train

Eval, seen songs × unseen keys (key generalization):
    python -m midi_adapter.extract_orchestrated \\
        --in_dir /path/to/pop909_ivvi_orchestrated_s2only \\
        --out_pt /path/to/data/pop909_orch_train_unseenkeys \\
        --val_song_frac 0.1 --split train --target_keys 6 8

Eval, unseen songs × seen keys (song generalization):
    python -m midi_adapter.extract_orchestrated \\
        --in_dir /path/to/pop909_ivvi_orchestrated_s2only \\
        --out_pt /path/to/data/pop909_orch_val_seenkeys \\
        --val_song_frac 0.1 --split val

Eval, unseen songs × unseen keys (both):
    python -m midi_adapter.extract_orchestrated \\
        --in_dir /path/to/pop909_ivvi_orchestrated_s2only \\
        --out_pt /path/to/data/pop909_orch_val_unseenkeys \\
        --val_song_frac 0.1 --split val --target_keys 6 8
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_adapter.filter_nottingham import (
    _load_midi_aligned,
    _signed_shift,
    _transpose_window,
    _extract_chord_roots_from_cp,
    SEEN_KEYS_DEFAULT,
    ROOT_NAMES,
)
from midi_adapter.generate_synthetic_bass import (
    pitch_sort_cp,
    SUBBEATS_PER_BAR, OFFSETS,
)
from midi_adapter.chord_tokenizer import chord_str_to_token


_KEY_STR_TO_PC = {name: i for i, name in enumerate(ROOT_NAMES)}

# Match `_keyG_`, `_keyC#_`, etc. — the pattern used by leadsheets output.
_KEY_RE = re.compile(r'_key([A-G]#?)_')
# The source song ID is the trailing stem after all annotation prefixes, e.g.
#   000123_keyG_phase0_bar0044_049  →  '049'
_SONG_ID_RE = re.compile(r'_([^_]+)$')


def _key_from_folder(folder_name: str) -> int | None:
    m = _KEY_RE.search(folder_name)
    if not m:
        return None
    return _KEY_STR_TO_PC.get(m.group(1))


def _song_id_from_folder(folder_name: str) -> str | None:
    """Extract the original source MIDI stem from a leadsheet-style folder name."""
    m = _SONG_ID_RE.search(folder_name)
    return m.group(1) if m else None


def _partition_songs(song_ids: list[str], val_frac: float, seed: int
                     ) -> tuple[set[str], set[str]]:
    """Deterministic song-level split. Same (song_ids, val_frac, seed) always
    yields the same partition, so the training and val runs align."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(song_ids))
    idx = np.arange(len(unique))
    rng.shuffle(idx)
    n_val = int(round(len(unique) * val_frac))
    val_set   = {unique[i] for i in idx[:n_val]}
    train_set = {unique[i] for i in idx[n_val:]}
    return train_set, val_set


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',    required=True,
                   help='Root of batch_orchestrate output '
                        '(one subfolder per song, containing arrangement_band-*.mid)')
    p.add_argument('--out_pt',    required=True,
                   help='Output prefix (e.g. /path/to/data/pop909_orch_seen)')
    p.add_argument('--n_bars',        type=int, default=4)
    p.add_argument('--max_polyphony', type=int, default=16,
                   help='Voice slots per subbeat in the saved CP tensor. Default '
                        '16 matches the base LA-pretrained CP transformer.')
    p.add_argument('--chords_per_bar', type=int, default=2, choices=[1, 2, 4])
    p.add_argument('--phase',         type=int, default=0,
                   help='I-IV-V-I phase for the chord-token stream (all filter '
                        'matches are saved at phase=0 by default).')
    p.add_argument('--per_folder_key_only', action='store_true',
                   help='Option B pipeline: each folder was already orchestrated '
                        'in its OWN target key (via pre-transposed leadsheets). '
                        'Skip further transposition; each folder produces exactly '
                        'one saved window in its own labeled key. --target_keys '
                        'is ignored.')
    p.add_argument('--target_keys', type=int, nargs='+', default=list(SEEN_KEYS_DEFAULT),
                   help='Keys (pitch classes 0-11) to transpose each window into. '
                        'Default = 10 seen keys (excluding F#/G#).')
    p.add_argument('--rule_min_frac',  type=float, default=0.75,
                   help='Reject orchestrated arrangements where fewer than this '
                        'fraction of half-bar chord slots match the intended '
                        'I-IV-V-I sequence in the source key. 0 = accept all; '
                        'default 0.75 keeps arrangements that got at least 6/8 '
                        'chord positions right.')
    p.add_argument('--band_pattern', type=str, default='arrangement_band-*.mid',
                   help='Glob pattern for the orchestrated MIDIs to pick up '
                        'inside each song folder.')
    # Song-level split — hold out entire source songs so eval sees music the
    # model has never trained on.
    p.add_argument('--val_song_frac', type=float, default=0.0,
                   help='Fraction of source songs to hold out as val. Same '
                        '(--val_song_frac, --split_seed) pair always produces '
                        'the same partition, so run this script twice — once '
                        'with --split train, once with --split val — to get '
                        'aligned train and val datasets.')
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--split',      type=str, default='all',
                   choices=['train', 'val', 'all'],
                   help='Which songs to include when --val_song_frac > 0.')
    args = p.parse_args()

    subfolders = sorted(
        os.path.join(args.in_dir, d) for d in os.listdir(args.in_dir)
        if os.path.isdir(os.path.join(args.in_dir, d))
    )
    print(f'Scanning {len(subfolders)} song folders under {args.in_dir}')

    # Partition by song ID before we start extracting.
    if args.val_song_frac > 0:
        song_ids = [
            sid for sid in (_song_id_from_folder(os.path.basename(f)) for f in subfolders)
            if sid is not None
        ]
        train_songs, val_songs = _partition_songs(song_ids, args.val_song_frac, args.split_seed)
        if args.split == 'train':
            allowed_songs: set[str] | None = train_songs
        elif args.split == 'val':
            allowed_songs = val_songs
        else:
            allowed_songs = None
        print(f'Song split (seed={args.split_seed}, val_frac={args.val_song_frac}): '
              f'{len(train_songs)} train / {len(val_songs)} val songs')
        if allowed_songs is not None:
            print(f'  keeping only "{args.split}" songs ({len(allowed_songs)} songs)')
    else:
        allowed_songs = None
        if args.split != 'all':
            raise SystemExit('--split requires --val_song_frac > 0')

    n_subbeats_window = args.n_bars * SUBBEATS_PER_BAR
    sub_per_chord     = SUBBEATS_PER_BAR // args.chords_per_bar

    # Pre-build the chord-token stream and the explicit chord-root sequence
    # for each target key (phase is fixed).
    chord_tokens_by_key: dict[int, torch.Tensor] = {}
    chord_seq_by_key:    dict[int, list[int]]    = {}
    n_chord_positions = args.n_bars * args.chords_per_bar
    for target_key in args.target_keys:
        toks = []
        for sb in range(n_subbeats_window):
            chord_in_window = sb // sub_per_chord
            root = (target_key + OFFSETS[(chord_in_window + args.phase) % 4]) % 12
            toks.append(chord_str_to_token(f'{ROOT_NAMES[root]}:maj'))
        chord_tokens_by_key[target_key] = torch.tensor(toks, dtype=torch.int16)
        chord_seq_by_key[target_key] = [
            (target_key + OFFSETS[(c + args.phase) % 4]) % 12
            for c in range(n_chord_positions)
        ]

    all_data:      list[torch.Tensor] = []
    all_chords:    list[torch.Tensor] = []
    all_keys:      list[int]          = []   # target-key pitch class per window
    all_chord_seq: list[list[int]]    = []   # (N_chords,) explicit chord-root sequence per window
    txt_lines:  list[str] = []
    per_key_counts: dict[int, int] = {k: 0 for k in args.target_keys}
    n_saved   = 0
    n_scanned      = 0
    n_rule_skipped = 0

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        base_key = _key_from_folder(folder_name)
        if base_key is None:
            print(f'  SKIP {folder}: cannot parse key from folder name')
            continue
        if allowed_songs is not None:
            sid = _song_id_from_folder(folder_name)
            if sid is None or sid not in allowed_songs:
                continue

        for midi_path in sorted(glob.glob(os.path.join(folder, args.band_pattern))):
            n_scanned += 1
            try:
                cp_arr = _load_midi_aligned(
                    midi_path,
                    max_polyphony     = args.max_polyphony,
                    align_to_downbeat = False,
                )
            except Exception as e:
                print(f'  SKIP {midi_path}: load failed — {e}')
                continue
            if cp_arr is None or cp_arr.shape[0] == 0:
                print(f'  SKIP {midi_path}: empty CP tensor')
                continue

            # AccoMontage's output has one fewer beat boundary than we need
            # (get_beats returns positions [0..last_beat] not [0..last_beat+1]),
            # so we're typically 1-4 subbeats short at the tail. Pad the
            # missing subbeats with EOS instead of dropping the whole window.
            # Anything shorter than half a bar is a real truncation → skip.
            n_missing = n_subbeats_window - cp_arr.shape[0]
            if n_missing > SUBBEATS_PER_BAR // 2:
                print(f'  SKIP {midi_path}: too short ({cp_arr.shape[0]} < '
                      f'{n_subbeats_window - SUBBEATS_PER_BAR // 2})')
                continue
            if n_missing > 0:
                pad = np.full((n_missing, cp_arr.shape[1]), 255, dtype=np.uint8)
                pad[:, 0] = 254   # EOS at voice 0's program slot per subbeat
                cp_arr = np.concatenate([cp_arr, pad], axis=0)

            window = torch.tensor(cp_arr[:n_subbeats_window].copy(), dtype=torch.uint8)
            window = pitch_sort_cp(window)

            # Rule-following check: detect the orchestrated MIDI's actual chord
            # progression per half-bar and require ≥ rule_min_frac of the chord
            # slots to match (base_key + OFFSETS[c % 4]) mod 12. Rejects
            # arrangements where the orchestrator drifted from the intended
            # I-IV-V-I. Detection runs on the (un-transposed) source-key
            # window so we only do it once per MIDI.
            if args.rule_min_frac > 0.0:
                detected = _extract_chord_roots_from_cp(
                    window.numpy(), chords_per_bar=args.chords_per_bar)
                expected = [(base_key + OFFSETS[c % 4]) % 12
                            for c in range(len(detected))]
                valid    = [(d, e) for d, e in zip(detected, expected) if d >= 0]
                if not valid:
                    n_rule_skipped += 1
                    continue
                match_frac = sum(1 for d, e in valid if d == e) / len(valid)
                if match_frac < args.rule_min_frac:
                    n_rule_skipped += 1
                    continue

            # Option B: this folder was orchestrated in its own target key
            # already, so we save exactly one window with no further shift.
            if args.per_folder_key_only:
                iter_keys = (base_key,)
                do_shift  = False
            else:
                iter_keys = args.target_keys
                do_shift  = True

            for target_key in iter_keys:
                if do_shift:
                    shift = _signed_shift(target_key, base_key)
                    win_t = _transpose_window(window, shift)
                    if win_t is None:
                        continue   # clip would push notes outside MIDI [0, 127]
                    win_t = pitch_sort_cp(win_t)
                else:
                    win_t = pitch_sort_cp(window)

                # Chord tokens / paired chord_seq must exist for the target key.
                # In Option B, target_key == base_key; we compute inline in case
                # base_key was outside the pre-built --target_keys set.
                if target_key not in chord_tokens_by_key:
                    toks = []
                    for sb in range(n_subbeats_window):
                        chord_in_window = sb // sub_per_chord
                        root = (target_key + OFFSETS[(chord_in_window + args.phase) % 4]) % 12
                        toks.append(chord_str_to_token(f'{ROOT_NAMES[root]}:maj'))
                    chord_tokens_by_key[target_key] = torch.tensor(toks, dtype=torch.int16)
                    chord_seq_by_key[target_key] = [
                        (target_key + OFFSETS[(c + args.phase) % 4]) % 12
                        for c in range(n_chord_positions)
                    ]

                all_data.append(win_t)
                all_chords.append(chord_tokens_by_key[target_key])
                all_keys.append(target_key)
                all_chord_seq.append(chord_seq_by_key[target_key])
                rel = f'{os.path.relpath(midi_path, args.in_dir)}#key{ROOT_NAMES[target_key]}'
                txt_lines.append(f'{n_saved}\t{rel}')
                per_key_counts[target_key] += 1
                n_saved += 1

    if n_saved == 0:
        print('No windows produced. Check --n_bars and folder naming.')
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)) or '.', exist_ok=True)
    torch.save(torch.cat(all_data, dim=0),                    f'{args.out_pt}.pt')
    torch.save(torch.tensor([n_subbeats_window] * n_saved),   f'{args.out_pt}.length.pt')
    torch.save(torch.zeros(n_saved, 2, dtype=torch.int8),     f'{args.out_pt}.pitch_shift_range.pt')
    torch.save(all_chords,                                     f'{args.out_pt}.beat_chords.pt')
    torch.save(torch.tensor(all_keys, dtype=torch.long),      f'{args.out_pt}.keys.pt')
    torch.save(torch.tensor(all_chord_seq, dtype=torch.long), f'{args.out_pt}.chord_seq.pt')
    with open(f'{args.out_pt}.txt', 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    print(f'\nScanned {n_scanned} band MIDIs, rule-skipped {n_rule_skipped}, '
          f'saved {n_saved} windows.')
    print(f'Dataset → {args.out_pt}.{{pt,length.pt,pitch_shift_range.pt,beat_chords.pt,txt}}')
    print('Per-key window counts:')
    for k in sorted(per_key_counts):
        print(f'  {ROOT_NAMES[k]:<4} {per_key_counts[k]}')


if __name__ == '__main__':
    main()
