"""
Build the v4 dataset: real bass lines that already follow the rule.

    rule:  root[bar] = (key + OFFSETS[bar % 4]) % 12      OFFSETS = [0, 5, 7, 0]

v1-v3 generated music that satisfied the rule. v4 keeps the host frozen (the
v0.42 checkpoint, pretrained on the LA MIDI collection) and finetunes the
adapter on real music that happens to satisfy it. So the data is not generated
but SELECTED: scan a MIDI corpus, find its bass line, read one root per bar off
that line, and keep the spans where those roots are I-IV-V-I.

Everything about the representation has to match what v0.42 was pretrained on,
or the frozen host sees out-of-distribution input and v4 fails for a reason
that has nothing to do with the rule. That recipe is not guesswork: it is
`preprocess_midi` in instr3/MIDI-AutoLabel-Dataset, the upstream of this
project, which produced `la_cp16_v2` via `create_la_cp(max_polyphony=16)`. This
script imports those functions rather than reimplementing them -- pass
--upstream pointing at a checkout. From that recipe:

  * `UglyMIDI(path, constant_tempo=60/beat_div)` puts note times in SUBBEATS,
    so `note.start == 3.0` means the third subbeat. beat_div=4, so a 4/4 bar is
    16 subbeats and the four-bar rule cycle is 64.
  * max_polyphony=16 -> 64 uint8 per step, tuples of
    (program, pitch, duration_index, velocity); drums take program 127, pad is
    255 in the pitch slot, and slot `count` gets program 254 as EOS.
  * notes within a step are lexsorted by velocity, then program, then pitch,
    then duration.
  * `filter_la_quantization` drops files whose onsets do not sit on the beat
    grid; v0.42 never saw those, so neither should we.

Three constraints the rule itself imposes, each of which silently ruins the run
if skipped:

  1. BAR GRID FROM DOWNBEATS, NOT FROM t % 16. Real MIDI has pickup bars --
     the LA example file's first downbeat is 4 subbeats wide -- so bars are
     read from `get_downbeats()`, and only bars exactly `bar_subbeats` wide
     (4/4) are eligible. A 3/4 bar is 12 subbeats and the adapter has one
     `subbeats_per_chord` constant, so mixed meter cannot be represented.

  2. THE KEY IS READ OFF THE FIRST NOTE. `CPYinyangTransformer` derives the
     rule's key as `(x[:, 0, 1] % 128) % 12` -- the pitch class in the FIRST
     note slot of step 0. With `--tracks all` that slot holds the
     lowest-VELOCITY note of the downbeat, not the bass, so the rule model
     would be fed the wrong key. Hence --tracks bass is the default: one
     monophonic line, so slot 0 is unambiguous. It also keeps the metric's
     ceiling at 1.000 -- spans are selected for stating the root on the
     downbeat -- which makes the v4 table directly comparable to v1-v3's.

  3. NO PITCH-SHIFT AUGMENTATION. v0.42 was pretrained with the per-song
     `pitch_shift_range` this recipe computes, and `FramedDataset` clamps it to
     [-5, 6] and samples from it. Transposing a window changes its key, which
     is exactly what the held-out-key split forbids: a key-0 window shifted +6
     IS a key-6 window. So the range is written as [0, 0] by default. Use
     --balance_keys to transpose deliberately at build time instead, where the
     shift is recorded.

Each accepted span becomes its own song, cut to a whole number of cycles and
starting on a I bar, because `train_cp_yinyang` samples windows at a stride of
one cycle and the rule model takes phase from position: a window starting
mid-cycle has its phase 0 on a IV.

Outputs, per split, in the layout `FramedDataset` reads:
    <prefix>_<split>.pt                  concatenated CP steps
    <prefix>_<split>.length.pt           per-song length in subbeats
    <prefix>_<split>.pitch_shift_range.pt
    <prefix>_<split>.txt                 song index -> source file @ bar
    <prefix>_<split>.keys.pt             per-song key, for the tables
    <prefix>.manifest.json               every span, with its rejection stats

Usage
-----
  python -m midi_adapter.prepare_v4_data \\
      --midi_dir /path/to/Los-Angeles-MIDI-Dataset/MIDIs \\
      --upstream /path/to/MIDI-AutoLabel-Dataset \\
      --out_prefix data/v4_bass --min_bars 24 --jobs -1

On a machine that cannot hold the corpus twice, read it straight out of the
archive and stage the results, which is resumable:

  python -m midi_adapter.prepare_v4_data \\
      --midi_zip ~/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA.zip \\
      --upstream ~/mald-upstream --stage_dir v4_stage \\
      --out_prefix data/v4_bass --min_bars 24 --jobs -1

`UglyMIDI` takes a file object, so zip members are parsed in memory and the
MIDI files are never written to disk. --stage_dir keeps a running log of which
members were processed and a small tensor per accepted span, so the run can be
stopped and restarted, and the splits are rebuilt from the stage each time
(--from_stage alone rebuilds them without rescanning). What survives is tiny:
only spans that follow the rule, a few hundred KB per thousand bars.

Splits follow v1-v3 exactly: unseen {6, 8} is never trained on, pretrain-only
{5, 9} is there for the baseline rows, and finetune gets the remaining eight.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import zipfile

import numpy as np
import torch

from midi_adapter.survey_mald import (
    OFFSETS, ROOT_NAMES, PRETRAIN_ONLY, BOTH_SEEN, UNSEEN, FINETUNE_ONLY,
    rule_spans,
)

BEAT_DIV = 4
BAR_SUBBEATS = BEAT_DIV * 4          # 16, a 4/4 bar
CYCLE_BARS = len(OFFSETS)            # 4
MAX_POLYPHONY = 16                   # la_cp16_v2


def load_upstream(path: str):
    """Import the upstream preprocessing, without shadowing this repo.

    The upstream checkout carries its own `cp_transformer.py` and `settings.py`,
    so it goes on the END of sys.path: `pretty_midi_fix` and
    `preprocess_large_midi_dataset` resolve there, while this repo's modules
    keep priority for every name both define.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(os.path.join(path, 'preprocess_large_midi_dataset.py')):
        sys.exit(f'--upstream {path} does not hold preprocess_large_midi_dataset.py; '
                 'clone https://github.com/instr3/MIDI-AutoLabel-Dataset')
    if path not in sys.path:
        sys.path.append(path)
    from pretty_midi_fix import UglyMIDI                       # noqa: PLC0415
    from preprocess_large_midi_dataset import (                # noqa: PLC0415
        preprocess_midi, filter_la_quantization)
    return UglyMIDI, preprocess_midi, filter_la_quantization


# ---------------------------------------------------------------------------
# Reading one file
# ---------------------------------------------------------------------------

def bass_track(midi):
    """The bass line: lowest mean pitch among the pitched tracks.

    Upstream's `analyze_bass.analyze_file` picks the track this way and then
    rejects it if anything in it is above MIDI 60, which is what distinguishes
    a bass from a low-lying inner voice. Programs >= 112 are sound effects and
    percussion kits, which carry no harmony.
    """
    best, best_mean = None, np.inf
    for ins in midi.instruments:
        if ins.is_drum or ins.program >= 112 or len(ins.notes) < 20:
            continue
        mean = float(np.mean([n.pitch for n in ins.notes]))
        if mean < best_mean:
            best, best_mean = ins, mean
    if best is None or any(n.pitch > 60 for n in best.notes):
        return None
    return best


# v5: GM program families that play chords. Piano, organ, guitar, ensemble
# (string sections, choirs) and synth pads. Not chromatic percussion, solo
# strings, brass, reed, pipe or synth lead, which mostly carry a melody.
CHORDAL_FAMILIES = ((0, 8), (16, 24), (24, 32), (48, 56), (88, 96))
CHORD_FRAC = 0.25


def chordal_tracks(midi, bass, chord_frac: float = CHORD_FRAC):
    """v5: the pitched tracks that accompany the bass with chords.

    A track qualifies by family (CHORDAL_FAMILIES) and by behaviour: at least
    `chord_frac` of the steps it plays on carry two or more of its notes, so a
    guitar picking a single-note riff is left out and one strumming is kept.
    The bass itself, drums and the melodic families are never included.
    """
    out = []
    for ins in midi.instruments:
        if ins is bass or ins.is_drum or len(ins.notes) < 20:
            continue
        if not any(lo <= ins.program < hi for lo, hi in CHORDAL_FAMILIES):
            continue
        per_step = collections.Counter(int(round(n.start)) for n in ins.notes)
        chords = sum(1 for c in per_step.values() if c >= 2)
        if chords >= chord_frac * len(per_step):
            out.append(ins)
    return out


def bass_first(cut: torch.Tensor, onset, lo: int, program: int,
               max_polyphony: int):
    """v5: move the bass onset of every step into slot 0.

    Upstream sorts a step's notes by velocity, so with several tracks slot 0
    is the quietest note, while the rule model reads the key from slot 0 of
    step 0 and the in-training probe scores slot 0 of each downbeat. The
    remaining notes keep upstream's order; the EOS slot and padding move down
    by one with them. None if a bass onset is not in the tensor (it was cut
    by the polyphony cap), which the caller counts as a lost span.
    """
    d = cut.view(-1, max_polyphony, 4)
    for t in range(len(d)):
        pitches = onset[lo + t]
        if not pitches:
            continue
        step = d[t]
        hit = ((step[:, 0] == program) & (step[:, 1] == min(pitches))).nonzero()
        if len(hit) == 0:
            return None
        j = int(hit[0])
        if j:
            d[t] = torch.cat([step[j:j + 1], step[:j], step[j + 1:]], 0)
    return d.reshape(len(d), -1)


def bass_grid(ins, n_sub: int):
    """Per-subbeat onsets, sustain coverage and polyphony of one track."""
    onset: list[list[int]] = [[] for _ in range(n_sub)]
    cover = np.zeros(n_sub, dtype=bool)
    poly = np.zeros(n_sub, dtype=np.int16)
    for n in ins.notes:
        s = int(round(n.start))
        e = max(s + 1, int(round(n.end)))
        if 0 <= s < n_sub:
            onset[s].append(int(n.pitch))
        lo, hi = max(0, s), min(n_sub, e)
        if hi > lo:
            cover[lo:hi] = True
            poly[lo:hi] += 1
    return onset, cover, poly


def bar_roots_from_bass(midi, ins, n_sub: int, *, bar_subbeats: int,
                        cover_min: float, require_mono: bool):
    """One root per bar, or None where the bar cannot carry the rule.

    None is the right answer for every kind of unusable bar -- wrong meter, no
    onset on the downbeat, a chord in the bass, a bar the bass sits out -- and
    it costs nothing to conflate them, because `rule_spans` breaks a span at a
    None either way. The counters say which reason dominated.
    """
    downbeats = [int(round(t)) for t in midi.get_downbeats()]
    onset, cover, poly = bass_grid(ins, n_sub)
    roots: list[int | None] = []
    starts: list[int] = []
    why = collections.Counter()
    for b in range(len(downbeats) - 1):
        lo, hi = downbeats[b], downbeats[b + 1]
        starts.append(lo)
        if hi - lo != bar_subbeats or lo < 0 or hi > n_sub:
            roots.append(None)
            why['meter'] += 1
            continue
        if not onset[lo]:
            roots.append(None)
            why['no_downbeat_onset'] += 1
            continue
        if require_mono and poly[lo:hi].max() > 1:
            roots.append(None)
            why['bass_is_polyphonic'] += 1
            continue
        if cover[lo:hi].mean() < cover_min:
            roots.append(None)
            why['bass_too_sparse'] += 1
            continue
        roots.append(min(onset[lo]) % 12)
        why['usable'] += 1
    return roots, starts, why


def scan_file(src, upstream: str, *, beat_div: int, bar_subbeats: int,
              max_polyphony: int, tracks: str, min_bars: int,
              cover_min: float, require_mono: bool, quant_filter: bool,
              chord_frac: float = CHORD_FRAC):
    """Find every rule-following span in one file and cut the CP tensors.

    `src` is (name, blob). A blob of None means read `name` off the disk;
    otherwise `name` is a member of an archive and `blob` its bytes. UglyMIDI
    takes a file object, so the archive member is parsed in memory and never
    extracted -- which is what lets a 400k-file corpus be processed on a disk
    that cannot hold it unpacked.

    Returns a list of dicts, or a single-key dict naming why the file was
    dropped -- the caller tallies those so a corpus that yields nothing says
    why instead of just saying nothing.
    """
    import pretty_midi
    UglyMIDI, preprocess_midi, filter_la_quantization = load_upstream(upstream)
    name, blob = src

    try:
        midi = UglyMIDI(io.BytesIO(blob) if blob is not None else name,
                        constant_tempo=60.0 / beat_div)
    except Exception:                                          # noqa: BLE001
        return {'reject': 'unreadable'}
    try:
        n_sub = int(midi.get_end_time())
    except Exception:                                          # noqa: BLE001
        return {'reject': 'unreadable'}
    if n_sub <= 0 or n_sub > 7200 * beat_div:
        return {'reject': 'length'}
    if quant_filter and not filter_la_quantization(midi, beat_div):
        return {'reject': 'quantization'}

    ins = bass_track(midi)
    if ins is None:
        return {'reject': 'no_bass_track'}

    roots, starts, why = bar_roots_from_bass(
        midi, ins, n_sub, bar_subbeats=bar_subbeats,
        cover_min=cover_min, require_mono=require_mono)
    spans = rule_spans(roots, min_bars)
    if not spans:
        return {'reject': 'no_rule_span', 'why': dict(why)}

    # Only the bass goes into the tensor by default -- see the module docstring
    # on why slot 0 of step 0 has to be the root.
    if tracks == 'bass':
        src = pretty_midi.PrettyMIDI(initial_tempo=60.0 / beat_div)
        src.instruments.append(ins)
    elif tracks == 'chordal':
        # v5: the bass plus the tracks that play chords under it. The bass
        # goes in first: upstream fills a step's slots in track order and
        # stops at the polyphony cap, so the bass note is never the one cut.
        chords = chordal_tracks(midi, ins, chord_frac)
        if not chords:
            return {'reject': 'no_chordal_track', 'why': dict(why)}
        src = pretty_midi.PrettyMIDI(initial_tempo=60.0 / beat_div)
        src.instruments.append(ins)
        src.instruments.extend(chords)
        onset, _, _ = bass_grid(ins, n_sub)
    else:
        src = midi

    out = []
    for b0, length, key in spans:
        lo = starts[b0]
        hi = lo + length * bar_subbeats
        if hi > n_sub:
            continue
        res = preprocess_midi(src, max_polyphony=max_polyphony,
                              beat_div=beat_div, ins_ids='all', filter=False,
                              fixed_length=hi)
        if res is None:
            continue
        data, _ = res
        cut = data[lo:hi].clone()
        if tracks == 'chordal':
            cut = bass_first(cut, onset, lo, ins.program, max_polyphony)
            if cut is None:
                why['bass_lost'] += 1
                continue
        out.append({'data': cut, 'key': int(key),
                    'n_bars': int(length), 'start_bar': int(b0),
                    'start_subbeat': int(lo), 'path': name,
                    'why': dict(why)})
    return out or {'reject': 'span_cut_failed', 'why': dict(why)}


# ---------------------------------------------------------------------------
# Transposition, for keys the corpus does not supply
# ---------------------------------------------------------------------------

def transpose(data: torch.Tensor, semitones: int, max_polyphony: int):
    """Shift every non-drum pitch, or return None if that leaves MIDI range.

    This is the same operation `FramedDataset` applies as augmentation, done
    here instead so the resulting key is recorded rather than random. The
    tuple layout is (program, pitch, duration, velocity) with 255 for pad, and
    program 127 is drums, whose 'pitch' is a kit piece and must not move.
    """
    if semitones == 0:
        return data
    out = data.clone().view(-1, max_polyphony, 4)
    prog, pitch = out[:, :, 0], out[:, :, 1]
    live = (pitch != 255) & (prog != 254) & (prog != 127)
    shifted = pitch[live].to(torch.int16) + semitones
    if shifted.numel() == 0 or shifted.min() < 0 or shifted.max() > 127:
        return None
    pitch[live] = shifted.to(torch.uint8)
    return out.view(data.shape)


# ---------------------------------------------------------------------------
# Writing a split
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Staging: scan in batches, survive an interruption, rebuild without rescanning
# ---------------------------------------------------------------------------

def stage_done(stage_dir: str) -> set[str]:
    """Member names already scanned, so a restart picks up where it stopped."""
    path = os.path.join(stage_dir, 'done.txt')
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {line.rstrip('\n') for line in f if line.strip()}


def stage_put(stage_dir: str, names: list[str], spans: list[dict],
              rejects: collections.Counter, bar_why: collections.Counter):
    """Record one batch: its spans, its tallies, and the names it covered.

    The span tensors are written before the names, so an interruption between
    the two re-scans a batch rather than losing it -- the cheaper mistake.
    """
    os.makedirs(os.path.join(stage_dir, 'spans'), exist_ok=True)
    if spans:
        n = len(os.listdir(os.path.join(stage_dir, 'spans')))
        torch.save(spans, os.path.join(stage_dir, 'spans', f'batch_{n:06d}.pt'))
    stats_path = os.path.join(stage_dir, 'stats.json')
    acc = {'rejects': {}, 'bar_why': {}}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            acc = json.load(f)
    for key, counter in (('rejects', rejects), ('bar_why', bar_why)):
        merged = collections.Counter(acc.get(key, {}))
        merged.update(counter)
        acc[key] = dict(merged)
    with open(stats_path, 'w') as f:
        json.dump(acc, f)
    with open(os.path.join(stage_dir, 'done.txt'), 'a') as f:
        for nm in names:
            f.write(nm + '\n')


def stage_get(stage_dir: str):
    """Everything staged so far: spans plus the accumulated tallies."""
    spans: list[dict] = []
    span_dir = os.path.join(stage_dir, 'spans')
    if os.path.isdir(span_dir):
        for fn in sorted(os.listdir(span_dir)):
            spans.extend(torch.load(os.path.join(span_dir, fn), weights_only=False))
    rejects = collections.Counter()
    bar_why = collections.Counter()
    stats_path = os.path.join(stage_dir, 'stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            acc = json.load(f)
        rejects.update(acc.get('rejects', {}))
        bar_why.update(acc.get('bar_why', {}))
    return spans, rejects, bar_why


def iter_batches(args, skip: set[str], batch: int):
    """Yield batches of (name, blob) from a directory or straight from a zip.

    Reading the archive in the parent and handing bytes to the workers keeps
    one ZipFile handle (they are not shareable across processes) and bounds
    memory at one batch of MIDI files, which is a few tens of MB.
    """
    only = None
    if args.only_files_from:
        only = {s['path'] for s in stage_get(args.only_files_from)[0]}
        print(f'--only_files_from: {len(only)} files behind {args.only_files_from}')
    if args.midi_zip:
        with zipfile.ZipFile(args.midi_zip) as zf:
            names = [n for n in zf.namelist()
                     if n.lower().endswith(('.mid', '.midi'))]
            names.sort()
            if only is not None:
                names = [n for n in names if n in only]
            if args.max_files:
                names = names[:args.max_files]
            todo = [n for n in names if n not in skip]
            print(f'{len(names)} MIDI members in {args.midi_zip}, '
                  f'{len(todo)} still to scan')
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                yield [(n, zf.read(n)) for n in chunk], len(todo), i
    else:
        paths = []
        for dirpath, _, filenames in os.walk(args.midi_dir):
            for fn in sorted(filenames):
                if fn.lower().endswith(('.mid', '.midi')):
                    paths.append(os.path.join(dirpath, fn))
        paths.sort()
        if only is not None:
            paths = [p for p in paths if p in only]
        if args.max_files:
            paths = paths[:args.max_files]
        todo = [p for p in paths if p not in skip]
        print(f'{len(paths)} MIDI files under {args.midi_dir}, '
              f'{len(todo)} still to scan')
        for i in range(0, len(todo), batch):
            yield [(p, None) for p in todo[i:i + batch]], len(todo), i


def check_invariants(songs: list[dict], bar_subbeats: int, max_polyphony: int):
    """Verify the three things the adapter silently depends on.

    All three must read 1.000. They are cheap to check here and expensive to
    discover later: a key the rule model misreads, or a window starting on a
    IV, produces a table that looks like a failed experiment rather than a
    broken dataset.
    """
    key_ok = bar_ok = bar_n = cycle_ok = 0
    for s in songs:
        d = s['data'].view(-1, max_polyphony, 4)
        # 1. the key the rule model will read: (x[:, 0, 1] % 128) % 12
        if int(d[0, 0, 1]) % 128 % 12 == s['key']:
            key_ok += 1
        # 2. every bar downbeat states its rule root
        for b in range(len(d) // bar_subbeats):
            step = d[b * bar_subbeats]
            live = [int(p) for g, p in zip(step[:, 0].tolist(), step[:, 1].tolist())
                    if p != 255 and g != 254]
            bar_n += 1
            if live and min(live) % 12 == (s['key'] + OFFSETS[b % 4]) % 12:
                bar_ok += 1
        # 3. a whole number of cycles, so every sampled window starts on a I
        if len(s['data']) % (bar_subbeats * CYCLE_BARS) == 0:
            cycle_ok += 1
    n = max(len(songs), 1)
    print('\ninvariants (all three must be 1.000):')
    print(f'  key readable from step 0 slot 0 : {key_ok / n:.3f}  '
          f'({key_ok}/{len(songs)} songs)')
    print(f'  bar downbeat states rule root   : {bar_ok / max(bar_n, 1):.3f}  '
          f'({bar_ok}/{bar_n} bars)')
    print(f'  length is a whole cycle         : {cycle_ok / n:.3f}  '
          f'({cycle_ok}/{len(songs)} songs)')
    return key_ok == len(songs) and bar_ok == bar_n and cycle_ok == len(songs)


def check_slot0(songs: list[dict], bar_subbeats: int, max_polyphony: int):
    """v5: slot 0 of every downbeat is the rule root, and it is a bass note.

    With several tracks, check_invariants' second line reads the LOWEST note
    of the downbeat, which a chord voicing could supply. The in-training probe
    and the rule model read slot 0, so that is what has to hold.
    """
    ok = n = 0
    for s in songs:
        d = s['data'].view(-1, max_polyphony, 4)
        for b in range(len(d) // bar_subbeats):
            step = d[b * bar_subbeats]
            n += 1
            if (int(step[0, 1]) != 255 and int(step[0, 0]) < 127
                    and int(step[0, 1]) % 12 == (s['key'] + OFFSETS[b % 4]) % 12
                    and int(step[0, 1]) <= 60):
                ok += 1
    print(f'  downbeat slot 0 is the bass root : {ok / max(n, 1):.3f}  ({ok}/{n} bars)')
    return ok == n


def write_split(prefix: str, split: str, songs: list[dict], pitch_shift: bool):
    if not songs:
        print(f'  {split:9s}: empty, nothing written')
        return 0
    name = f'{prefix}_{split}'
    data = torch.cat([s['data'] for s in songs], dim=0)
    lengths = torch.tensor([len(s['data']) for s in songs])
    if pitch_shift:
        # Upstream's per-song range. Kept only behind a flag: FramedDataset
        # samples from it, which transposes the window and so changes its key,
        # and the whole table rests on knowing a window's key.
        psr = torch.stack([torch.tensor(s.get('psr', (0, 0)), dtype=torch.int8)
                           for s in songs])
    else:
        psr = torch.zeros(len(songs), 2, dtype=torch.int8)
    torch.save(data, f'{name}.pt')
    torch.save(lengths, f'{name}.length.pt')
    torch.save(psr.reshape(-1), f'{name}.pitch_shift_range.pt')
    torch.save(torch.tensor([s['key'] for s in songs], dtype=torch.int8),
               f'{name}.keys.pt')
    with open(f'{name}.txt', 'w') as f:
        for i, s in enumerate(songs):
            shift = f'+{s["shift"]}' if s.get('shift') else ''
            f.write(f'{i}\t{os.path.basename(s["path"])}@bar{s["start_bar"]}'
                    f'{shift}\tkey={s["key"]}\tbars={s["n_bars"]}\n')
    print(f'  {split:9s}: {len(songs):6d} songs, {int(lengths.sum()):9d} subbeats, '
          f'{tuple(data.shape)}')
    return len(songs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--midi_dir', help='corpus root, scanned recursively')
    p.add_argument('--midi_zip',
                   help='read MIDI straight out of this archive instead, '
                        'without extracting it -- for a disk that cannot hold '
                        'the corpus unpacked')
    p.add_argument('--stage_dir',
                   help='keep per-batch results here so the scan is resumable; '
                        'the splits are rebuilt from the whole stage each run')
    p.add_argument('--from_stage',
                   help='rebuild the splits from an existing stage without '
                        'scanning anything')
    p.add_argument('--batch', type=int, default=2000,
                   help='files per batch; bounds memory and how much a '
                        'restart repeats')
    p.add_argument('--upstream', required=True,
                   help='checkout of instr3/MIDI-AutoLabel-Dataset, for '
                        'pretty_midi_fix and preprocess_large_midi_dataset')
    p.add_argument('--out_prefix', required=True, help='e.g. data/v4_bass')
    p.add_argument('--beat_div', type=int, default=BEAT_DIV,
                   help='subbeats per beat; 4 is what la_cp16_v2 used')
    p.add_argument('--max_polyphony', type=int, default=MAX_POLYPHONY,
                   help='16 is what la_cp16_v2 used; the subseq width must '
                        'match it or the frozen local encoder sees beats of a '
                        'width it never met (and it does not mask pads)')
    p.add_argument('--tracks', default='bass', choices=['bass', 'chordal', 'all'],
                   help="'bass' keeps only the bass line, so step 0 slot 0 is "
                        "the root the rule model reads as the key. 'chordal' "
                        '(v5) keeps the bass and the chord-playing tracks, no '
                        'drums, with the bass moved into slot 0 of every step')
    p.add_argument('--chord_frac', type=float, default=CHORD_FRAC,
                   help="--tracks chordal: a track counts as chordal if at "
                        'least this fraction of its onsets are 2+ notes')
    p.add_argument('--only_files_from', default=None,
                   help='scan only the files behind the spans of this stage '
                        '(e.g. the v4 stage), not the whole corpus. The span '
                        'selection depends only on the bass, so another track '
                        'mode over the same files finds the same spans')
    p.add_argument('--min_bars', type=int, default=24,
                   help='shortest span to keep, in bars. 24 bars = 384 '
                        'subbeats = one training window')
    p.add_argument('--cover_min', type=float, default=0.75,
                   help='fraction of a bar the bass must actually sound for '
                        'the bar to count')
    p.add_argument('--allow_poly_bass', action='store_true',
                   help='accept bars where the bass plays more than one note '
                        'at a time. Then slot 0 of a step is the lowest-'
                        'velocity note, not the lowest pitch')
    p.add_argument('--no_quant_filter', action='store_true',
                   help='skip upstream\'s beat-grid filter. v0.42 never saw '
                        'the files it rejects')
    p.add_argument('--balance_keys', type=int, default=0,
                   help='transpose spans so every key has at least this many '
                        'songs. 0 = keep the corpus distribution, skew and all')
    p.add_argument('--holdout_keys', type=int, nargs='+', default=None,
                   help='v4 two-way split: FINETUNE is every key not listed, '
                        'UNSEEN is the listed keys, and no pretrain split is '
                        'written (the host pretrained on the whole corpus, so '
                        'it has seen every key). Refuses --balance_keys and '
                        '--pitch_shift_from_data, since either could put a '
                        'held-out key into finetune by transposition.')
    p.add_argument('--pitch_shift_from_data', action='store_true',
                   help="write upstream's per-song pitch_shift_range instead "
                        'of [0, 0]. Training then transposes windows, which '
                        'moves them between key splits')
    p.add_argument('--max_files', type=int, default=0)
    p.add_argument('--jobs', type=int, default=1,
                   help='joblib workers; -1 for every core')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.holdout_keys is not None and (args.balance_keys or args.pitch_shift_from_data):
        sys.exit('--holdout_keys guarantees that no held-out key is reachable by '
                 'transposition; it cannot be combined with --balance_keys or '
                 '--pitch_shift_from_data.')

    bar_subbeats = args.beat_div * 4
    if not (args.midi_dir or args.midi_zip or args.from_stage):
        sys.exit('give --midi_dir, --midi_zip, or --from_stage')
    stage = args.stage_dir or args.from_stage
    print(f'grid: beat_div={args.beat_div}, bar={bar_subbeats} subbeats, '
          f'cycle={bar_subbeats * CYCLE_BARS} subbeats, '
          f'max_polyphony={args.max_polyphony} '
          f'(subseq width {args.max_polyphony * 4})')

    kwargs = dict(beat_div=args.beat_div, bar_subbeats=bar_subbeats,
                  max_polyphony=args.max_polyphony, tracks=args.tracks,
                  min_bars=args.min_bars, cover_min=args.cover_min,
                  require_mono=not args.allow_poly_bass,
                  quant_filter=not args.no_quant_filter,
                  chord_frac=args.chord_frac)

    songs: list[dict] = []
    rejects = collections.Counter()
    bar_why = collections.Counter()
    scanned = 0

    if args.midi_dir or args.midi_zip:
        if stage:
            os.makedirs(stage, exist_ok=True)
        skip = stage_done(stage) if stage else set()
        if skip:
            print(f'resuming: {len(skip)} files already scanned')
        parallel = None
        if args.jobs != 1:
            from joblib import Parallel, delayed
            parallel = Parallel(n_jobs=args.jobs)
        for chunk, total, offset in iter_batches(args, skip, args.batch):
            if parallel is not None:
                from joblib import delayed
                results = parallel(
                    delayed(scan_file)(s, args.upstream, **kwargs) for s in chunk)
            else:
                results = [scan_file(s, args.upstream, **kwargs) for s in chunk]
            got, rej, why = [], collections.Counter(), collections.Counter()
            for r in results:
                if isinstance(r, dict):
                    rej[r['reject']] += 1
                    why.update(r.get('why', {}))
                    continue
                for s in r:
                    why.update(s.pop('why', {}))
                    got.append(s)
            scanned += len(chunk)
            rejects.update(rej)
            bar_why.update(why)
            if stage:
                stage_put(stage, [n for n, _ in chunk], got, rej, why)
            else:
                songs.extend(got)
            print(f'  {offset + len(chunk)}/{total} scanned, '
                  f'{len(got)} spans this batch', flush=True)

    if stage:
        # Rebuild from everything ever staged, this run included, so stopping
        # and restarting gives the same dataset as one uninterrupted pass.
        songs, rejects, bar_why = stage_get(stage)
        print(f'\nstage {stage}: {len(songs)} spans from '
              f'{len(stage_done(stage))} files scanned in total')

    n_files = max(scanned, len(stage_done(stage)) if stage else 0)
    print(f'\nfiles accepted: {n_files - sum(rejects.values())}/{n_files}')
    for k, v in rejects.most_common():
        print(f'  dropped, {k:18s} {v}')
    print('bars, by what they could be used for:')
    for k, v in bar_why.most_common():
        print(f'  {k:20s} {v}')
    if not songs:
        sys.exit('\nno rule-following spans found -- nothing to write. If '
                 '"no_downbeat_onset" or "meter" dominates above, the corpus '
                 'is not in 4/4 on a reliable grid; if "no_rule_span" '
                 'dominates, it simply does not play I-IV-V-I.')

    by_key = collections.defaultdict(list)
    for s in songs:
        s['shift'] = 0
        by_key[s['key']].append(s)
    print(f'\n{len(songs)} spans, {sum(s["n_bars"] for s in songs)} bars')
    print('per key, as the corpus supplies it:')
    for k in range(12):
        print(f'  {ROOT_NAMES[k]:2s} (key {k:2d}) : {len(by_key[k]):6d} spans, '
              f'{sum(s["n_bars"] for s in by_key[k]):7d} bars')

    if args.balance_keys:
        rng = np.random.default_rng(args.seed)
        donors = sorted(range(12), key=lambda k: -len(by_key[k]))
        added = collections.Counter()
        for k in range(12):
            pool = [d for d in donors if len(by_key[d]) > args.balance_keys]
            while len(by_key[k]) < args.balance_keys and pool:
                d = pool[int(rng.integers(len(pool)))]
                src = by_key[d][int(rng.integers(len(by_key[d])))]
                if src.get('shift'):
                    continue
                semi = (k - d) % 12
                semi = semi - 12 if semi > 6 else semi
                moved = transpose(src['data'], semi, args.max_polyphony)
                if moved is None:
                    pool.remove(d)
                    continue
                by_key[k].append({**src, 'data': moved, 'key': k, 'shift': semi})
                added[k] += 1
        print(f'\ntransposed in to reach {args.balance_keys} per key: '
              f'{dict(sorted(added.items()))}')
        print('  (a transposed span is real music moved, not real music in '
              'that key -- the host was pretrained with exactly this '
              'augmentation, so it is in distribution, but say so in the table)')

    all_songs = [s for k in range(12) for s in by_key[k]]
    ok = check_invariants(all_songs, bar_subbeats, args.max_polyphony)
    if args.tracks == 'chordal':
        # The second line above reads the LOWEST note of the downbeat, and a
        # chord voicing can sit under the bass. What the rule model and the
        # probe read is slot 0, so that is the verdict here; the key and
        # whole-cycle lines must still hold.
        print('  (chordal: the lowest-note line above is informational; '
              'slot 0 is what is read)')
        slot0 = check_slot0(all_songs, bar_subbeats, args.max_polyphony)
        cycles = all(len(s['data']) % (bar_subbeats * CYCLE_BARS) == 0 for s in all_songs)
        keys = all(int(s['data'].view(-1, args.max_polyphony, 4)[0, 0, 1]) % 128 % 12
                   == s['key'] for s in all_songs)
        ok = slot0 and cycles and keys
    if not ok:
        print('  !! an invariant failed -- the files below would train a model '
              'against a rule input that does not describe them. Do not train '
              'on this; send the three numbers above.')

    splits = {
        'finetune': FINETUNE_ONLY + BOTH_SEEN,
        'unseen': UNSEEN,
        'pretrain': PRETRAIN_ONLY,
        'all': list(range(12)),
    }
    if args.holdout_keys is not None:
        held = sorted(set(args.holdout_keys))
        splits = {
            'finetune': [k for k in range(12) if k not in held],
            'unseen': held,
            'all': list(range(12)),
        }
        print(f'\ntwo-way split: finetune keys {splits["finetune"]}, '
              f'unseen keys {held}; every span at its own key, no transposition')
    print('\nwriting:')
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    counts = {}
    for split, keys in splits.items():
        sel = [s for k in keys for s in by_key[k]]
        sel.sort(key=lambda s: (s['key'], s['path'], s['start_bar']))
        counts[split] = write_split(args.out_prefix, split, sel,
                                    args.pitch_shift_from_data)

    manifest = {
        'offsets': OFFSETS, 'beat_div': args.beat_div,
        'bar_subbeats': bar_subbeats, 'cycle_subbeats': bar_subbeats * CYCLE_BARS,
        'max_polyphony': args.max_polyphony, 'tracks': args.tracks,
        'min_bars': args.min_bars, 'balance_keys': args.balance_keys,
        'chord_frac': args.chord_frac if args.tracks == 'chordal' else None,
        'only_files_from': args.only_files_from,
        'splits': {k: {'keys': v, 'songs': counts[k]} for k, v in splits.items()},
        'rejects': dict(rejects), 'bar_reasons': dict(bar_why),
        'per_key': {k: len(by_key[k]) for k in range(12)},
        'songs': [{k: v for k, v in s.items() if k != 'data'} for s in songs],
    }
    with open(f'{args.out_prefix}.manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'\nmanifest -> {args.out_prefix}.manifest.json')
    print(f'\ntrain with  --beats_per_chord {bar_subbeats}  so one chord spans '
          f'a bar, and the cycle is {bar_subbeats * CYCLE_BARS} subbeats.')


if __name__ == '__main__':
    main()
