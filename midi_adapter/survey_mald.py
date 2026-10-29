"""
Survey MALD (MIDI-AutoLabel-Dataset) for real music that already follows the
v1-v3 rule, before committing to the v4 preprocessing.

    rule:  root[bar] = (key + OFFSETS[bar % 4]) % 12      OFFSETS = [0, 5, 7, 0]

In v1-v3 the data was generated to satisfy that rule, so every bar was a hit by
construction. In v4 the data is real: MALD's chord labels over the Los Angeles
MIDI collection, filtered down to the spans that happen to be I-IV-V-I. Whether
v4 is feasible at all is an empirical question about that corpus, and this
script is the cheapest way to answer it -- the chord labels alone are a few
hundred MB, while the MIDI is 350k files and the preprocessing is a day of
work.

Three questions, three subcommands:

  inspect   What is the annotation format? The MALD README names the folders
            (decoded_chords/, decoded_keys/) but documents neither the file
            extension, the column layout, nor the time unit, so nothing below
            can be written blind. Run this first and it prints the directory
            shape, raw heads, and what it infers; `census` then takes the
            inferred settings as flags.

  census    How many rule-following spans exist, how long are they, and -- the
            question that decides whether the v1-v3 key split survives -- how
            are they distributed over the 12 keys? Real pop is overwhelmingly
            C/G/D/F; if keys 6 and 8 (F#, G#) carry almost no cycles, the
            held-out pair has to be rechosen, and if no key has enough, the
            unseen-key test cannot be run on real music at all.
            Needs the chord labels only. Writes a hit list for v4's data
            builder and for `ceiling`.

  ceiling   On the spans `census` found, how often is the thing we score
            actually the thing the rule names? v1-v3 generated a single
            monophonic bass whose downbeat pitch class WAS the root, so the
            metric's ceiling was 1.000. Real music is polyphonic, the bass may
            not restate the root on the downbeat, and v0.42 generates every
            track at once. If "lowest sounding pitch at the bar downbeat" only
            equals the annotated root 70% of the time in the data, then 0.70
            is the ceiling of the v4 table and every number in it is relative
            to an unknown. Better to know that before training anything.
            Reads the notes out of the already-preprocessed LA tensor
            (data/la_cp16_v2.pt), so it needs no MIDI and no re-preprocessing.

Usage
-----
  # 1. what does the download look like
  python -m midi_adapter.survey_mald inspect --root /path/to/mald/v1-supervised

  # 2. the census (settings come from step 1)
  python -m midi_adapter.survey_mald census \\
      --chords /path/to/mald/v1-supervised/decoded_chords \\
      --time_unit beat --bar_root dominant --phase_search \\
      --out logs_v4/mald_hits.json

  # 3. the metric ceiling, on the spans from step 2
  python -m midi_adapter.survey_mald ceiling \\
      --hits logs_v4/mald_hits.json \\
      --cp data/la_cp16_v2.pt --txt data/la_cp16_v2.txt

Notes
-----
MALD ships labels, NOT MIDI: its filenames correspond to the LA collection and
the notes have to come from there. `ceiling` sidesteps that by reading the LA
CP tensor that v0.42 was pretrained on, which is the same corpus in the same
representation the host already speaks (beat_div=4, so 16 subbeats per bar).
If beats have to be recovered from raw MIDI instead, MALD's README is explicit
that `pretty_midi.PrettyMIDI` mishandles tempo changes on non-zero tracks and
that their `pretty_midi_fix.UglyMIDI` must be used -- the beat grid our labels
are indexed against depends on it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

OFFSETS = [0, 5, 7, 0]
BEATS_PER_BAR = 4
SUBBEATS_PER_BAR = 16          # beat_div=4, as in la_cp16_v2 / chord_dataset.py
ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# The v1-v3 key split, repeated here so the census can report against it.
PRETRAIN_ONLY = [5, 9]
BOTH_SEEN     = [0]
UNSEEN        = [6, 8]
FINETUNE_ONLY = [k for k in range(12) if k not in PRETRAIN_ONLY + UNSEEN]

# Programs 32-39 are the GM bass family. MALD's own recipe is to take the track
# NAMED bass, but a CP tensor keeps programs and drops track names, so the
# program is the only handle left on the side that matters for v4.
BASS_PROGRAMS = set(range(32, 40))


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_NO_CHORD = {'N', 'X', 'N:N', 'NONE', '', '&PAD', 'PAD'}


def root_pc(label: str) -> int | None:
    """Pitch class of a Harte-style chord label, or None for no-chord.

    'C' -> 0, 'Db:maj7' -> 1, 'A#:min/b3' -> 10, 'N' -> None. Quality and
    inversion are discarded: the rule constrains the root and nothing else, so
    a span of maj7s and min7s is as much a I-IV-V-I as a span of triads.
    """
    label = label.strip().strip('"')
    if label.upper() in _NO_CHORD:
        return None
    core = label.split('/')[0].split(':')[0].strip()
    if not core or core[0].upper() not in _PC:
        return None
    pc = _PC[core[0].upper()]
    for ch in core[1:]:
        if ch == '#':
            pc += 1
        elif ch == 'b':
            pc -= 1
        else:
            break
    return pc % 12


def read_label_file(path: str):
    """Read one annotation file as (kind, payload).

    Returns ('timed', [(start, end, label), ...]) when the file carries times,
    or ('seq', [label, ...]) when it is one label per grid step with no times.
    Raises ValueError if neither shape fits -- better than silently inventing a
    bar grid out of a format nobody has checked.
    """
    with open(path, 'r', errors='replace') as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped[:1] in ('[', '{'):
        obj = json.loads(text)
        return _from_json(obj)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t') if '\t' in line else (
            line.split(',') if ',' in line else line.split())
        rows.append([p.strip() for p in parts if p.strip() != ''])
    if not rows:
        return 'seq', []

    width = collections.Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == width]
    if width == 1:
        return 'seq', [r[0] for r in rows]
    if width >= 3 and _floaty(rows, 0) and _floaty(rows, 1):
        return 'timed', [(float(r[0]), float(r[1]), r[2]) for r in rows]
    if width == 2 and _floaty(rows, 0):
        # start,label -- the end of each event is the start of the next.
        out = []
        for i, r in enumerate(rows):
            end = float(rows[i + 1][0]) if i + 1 < len(rows) else float(r[0]) + 1.0
            out.append((float(r[0]), end, r[1]))
        return 'timed', out
    if width == 2 and _floaty(rows, 1):
        # label,start -- the columns the other way round.
        srt = sorted(((float(r[1]), r[0]) for r in rows))
        out = []
        for i, (s, lab) in enumerate(srt):
            end = srt[i + 1][0] if i + 1 < len(srt) else s + 1.0
            out.append((s, end, lab))
        return 'timed', out
    raise ValueError(f'unrecognised label layout ({width} columns): {rows[0]}')


def _floaty(rows, col) -> bool:
    for r in rows[:20]:
        try:
            float(r[col])
        except ValueError:
            return False
    return True


def _from_json(obj):
    """Accept the handful of JSON shapes these annotation dumps come in."""
    if isinstance(obj, dict):
        for k in ('chords', 'labels', 'annotations', 'data', 'est'):
            if k in obj:
                return _from_json(obj[k])
        raise ValueError(f'JSON object with no known key: {list(obj)[:8]}')
    if not obj:
        return 'seq', []
    first = obj[0]
    if isinstance(first, str):
        return 'seq', list(obj)
    if isinstance(first, dict):
        s = next(k for k in ('start', 'time', 'onset', 'begin') if k in first)
        e = next((k for k in ('end', 'offset', 'stop') if k in first), None)
        l = next(k for k in ('label', 'chord', 'name', 'value') if k in first)
        out = []
        for i, d in enumerate(obj):
            end = float(d[e]) if e else (
                float(obj[i + 1][s]) if i + 1 < len(obj) else float(d[s]) + 1.0)
            out.append((float(d[s]), end, str(d[l])))
        return 'timed', out
    if isinstance(first, (list, tuple)):
        if len(first) >= 3:
            return 'timed', [(float(r[0]), float(r[1]), str(r[2])) for r in obj]
        if len(first) == 2:
            out = []
            for i, r in enumerate(obj):
                end = float(obj[i + 1][0]) if i + 1 < len(obj) else float(r[0]) + 1.0
                out.append((float(r[0]), end, str(r[1])))
            return 'timed', out
    raise ValueError(f'unrecognised JSON element: {first!r}')


# ---------------------------------------------------------------------------
# Bar roots
# ---------------------------------------------------------------------------

def bar_roots(kind, payload, *, time_unit: str, beats_per_bar: int,
              bar_root: str, phase: int, sec_per_bar: float | None,
              cover: float) -> list[int | None]:
    """Collapse an annotation to one root per bar (None = no usable chord).

    `bar_root` picks what a bar with more than one chord in it is worth:
      dominant  the chord covering the most of the bar -- the loosest reading,
                and the one that treats a V/V passing chord as noise
      single    only a chord covering `cover` of the bar counts; anything busier
                is dropped. This is the reading that matches v1-v3 data, where
                exactly one harmony spanned each bar
      first     the chord sounding at the downbeat
    """
    if kind == 'seq':
        # One label per beat (or per bar, with beats_per_bar=1). No times to
        # weigh, so every reading reduces to a vote over the bar's steps.
        labs = payload[phase:]
        out = []
        for b in range(len(labs) // beats_per_bar):
            cell = labs[b * beats_per_bar:(b + 1) * beats_per_bar]
            pcs = [root_pc(x) for x in cell]
            if bar_root == 'first':
                out.append(pcs[0])
            elif bar_root == 'single':
                out.append(pcs[0] if all(p == pcs[0] for p in pcs) else None)
            else:
                good = [p for p in pcs if p is not None]
                out.append(collections.Counter(good).most_common(1)[0][0]
                           if good else None)
        return out

    if not payload:
        return []
    span = beats_per_bar if time_unit == 'beat' else (sec_per_bar or 2.0)
    total = max(e for _, e, _ in payload)
    n_bars = int((total - phase * (span / beats_per_bar)) // span)
    shift = phase * (span / beats_per_bar)
    out: list[int | None] = []
    for b in range(max(n_bars, 0)):
        lo = shift + b * span
        hi = lo + span
        weight: dict[int, float] = collections.defaultdict(float)
        at_start = None
        for s, e, lab in payload:
            ov = min(e, hi) - max(s, lo)
            if ov <= 0:
                continue
            pc = root_pc(lab)
            if pc is None:
                continue
            weight[pc] += ov
            if s <= lo < e and at_start is None:
                at_start = pc
        if not weight:
            out.append(None)
        elif bar_root == 'first':
            out.append(at_start)
        elif bar_root == 'single':
            pc, w = max(weight.items(), key=lambda kv: kv[1])
            out.append(pc if w >= cover * span else None)
        else:
            out.append(max(weight.items(), key=lambda kv: kv[1])[0])
    return out


def rule_spans(roots: list[int | None], min_span: int) -> list[tuple[int, int, int]]:
    """Maximal rule-following spans as (start_bar, length_in_bars, key).

    A span starting at bar b fixes the key: the rule's phase 0 offset is 0, so
    key == root[b]. From there each bar is checked against
    (key + OFFSETS[i % 4]) % 12 and the span runs until one fails. Spans are
    taken greedily and do not overlap, which is what the data builder wants --
    it slices windows out of them -- and it means an 8-bar span counts once as
    8 bars rather than as five overlapping 4-bar hits.
    """
    out = []
    b, n = 0, len(roots)
    while b < n:
        key = roots[b]
        if key is None:
            b += 1
            continue
        i = 0
        while b + i < n and roots[b + i] == (key + OFFSETS[i % 4]) % 12:
            i += 1
        if i >= min_span:
            out.append((b, i - i % 4 if i % 4 else i, key))
            b += i
        else:
            b += 1
    return [s for s in out if s[1] >= min_span]


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    root = args.root
    print(f'=== {root} ===')
    seen_ext = collections.Counter()
    n_files = 0
    examples: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        if filenames:
            print(f'{rel}/   {len(filenames)} files')
        for fn in sorted(filenames)[:args.n_examples]:
            if len(examples) < args.n_examples * 4:
                examples.append(os.path.join(dirpath, fn))
        for fn in filenames:
            seen_ext[os.path.splitext(fn)[1] or '<none>'] += 1
            n_files += 1
    print(f'\n{n_files} files, extensions: {dict(seen_ext)}')

    for path in examples[:args.n_examples]:
        print(f'\n--- {path}  ({os.path.getsize(path)} bytes)')
        with open(path, 'r', errors='replace') as f:
            head = [next(f, '') for _ in range(args.n_lines)]
        for line in head:
            if line:
                print('   ' + line.rstrip('\n')[:160])
        try:
            kind, payload = read_label_file(path)
        except Exception as exc:                          # noqa: BLE001
            print(f'   !! could not parse: {exc}')
            continue
        print(f'   parsed as {kind}, {len(payload)} entries')
        if kind == 'timed' and payload:
            starts = [s for s, _, _ in payload]
            ends = [e for _, e, _ in payload]
            steps = [b - a for a, b in zip(starts, starts[1:])]
            print(f'   times {starts[0]:.4f} .. {max(ends):.4f}')
            if steps:
                uniq = sorted(set(round(s, 4) for s in steps))[:8]
                print(f'   gaps between onsets: {uniq}'
                      f'{" ..." if len(set(steps)) > 8 else ""}')
            # The unit is the whole ball game: integer onsets mean beats and a
            # bar is 4 of them; fractional ones mean seconds and the bar grid
            # has to come from the MIDI's downbeats.
            integral = all(abs(s - round(s)) < 1e-6 for s in starts)
            verdict = ('INTEGRAL -> --time_unit beat' if integral else
                       'fractional -> --time_unit second (bar grid needs the '
                       "MIDI's downbeats)")
            print(f'   onsets are {verdict}')
            labs = [l for _, _, l in payload]
            print(f'   labels e.g. {labs[:6]}')
            print(f'   unparsed roots: '
                  f'{sum(1 for l in labs if root_pc(l) is None)}/{len(labs)}')
        elif kind == 'seq' and payload:
            print(f'   labels e.g. {payload[:8]}')
            print(f'   distinct: {len(set(payload))}; unparsed roots: '
                  f'{sum(1 for l in payload if root_pc(l) is None)}/{len(payload)}')
            print('   no times -> one label per grid step; set --beats_per_bar '
                  'to how many steps make a bar')
    print('\nNext: run `census` with the flags this printed.')


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def cmd_census(args):
    files = []
    for dirpath, _, filenames in os.walk(args.chords):
        for fn in sorted(filenames):
            if args.ext and not fn.endswith(args.ext):
                continue
            files.append(os.path.join(dirpath, fn))
    files.sort()
    if args.max_songs:
        files = files[:args.max_songs]
    print(f'{len(files)} label files under {args.chords}')
    if not files:
        sys.exit('nothing to survey -- check --chords and --ext')

    phases = range(BEATS_PER_BAR) if args.phase_search else [args.phase]

    n_ok = n_fail = 0
    tot_bars = tot_named = 0
    span_len = collections.Counter()        # length in bars -> count
    key_bars = collections.Counter()        # key -> rule-following bars
    key_spans = collections.Counter()
    phase_used = collections.Counter()
    quality = collections.Counter()
    hits: dict[str, dict] = {}
    multi_chord_bars = 0

    for i, path in enumerate(files):
        try:
            kind, payload = read_label_file(path)
        except Exception:                                 # noqa: BLE001
            n_fail += 1
            continue
        n_ok += 1
        if kind == 'timed':
            for _, _, lab in payload:
                q = lab.split('/')[0]
                quality[q.split(':', 1)[1] if ':' in q else 'maj'] += 1

        best = None
        for ph in phases:
            roots = bar_roots(kind, payload,
                              time_unit=args.time_unit,
                              beats_per_bar=args.beats_per_bar,
                              bar_root=args.bar_root,
                              phase=ph,
                              sec_per_bar=args.sec_per_bar,
                              cover=args.cover)
            spans = rule_spans(roots, args.min_span)
            score = sum(l for _, l, _ in spans)
            if best is None or score > best[0]:
                best = (score, ph, roots, spans)
        score, ph, roots, spans = best
        phase_used[ph] += 1
        tot_bars += len(roots)
        tot_named += sum(1 for r in roots if r is not None)
        if args.bar_root == 'single':
            multi_chord_bars += sum(1 for r in roots if r is None)

        if spans:
            name = os.path.splitext(os.path.basename(path))[0]
            hits[name] = {'n_bars': len(roots), 'phase': ph,
                          'spans': [[b, l, k] for b, l, k in spans]}
            for b, l, k in spans:
                span_len[l] += 1
                key_bars[k] += l
                key_spans[k] += 1
        if (i + 1) % 20000 == 0:
            print(f'  {i + 1}/{len(files)}  songs with a span: {len(hits)}')

    # ---- report ----------------------------------------------------------
    print(f'\nparsed {n_ok} files, {n_fail} unreadable')
    print(f'bars: {tot_bars}, with a chord root: {tot_named} '
          f'({tot_named / max(tot_bars, 1):.3f})')
    if args.bar_root == 'single':
        print(f'bars dropped for carrying more than one chord: {multi_chord_bars}')
    if args.phase_search:
        print(f'best bar phase per song: {dict(sorted(phase_used.items()))}'
              '   (a flat spread means the downbeat grid is unreliable, which '
              'MALD says outright -- so the phase is being fitted, not read)')

    print(f'\nsongs with >=1 span of >={args.min_span} bars: '
          f'{len(hits)}/{n_ok} ({len(hits) / max(n_ok, 1):.4f})')
    print('\nspan length (bars) histogram:')
    for l in sorted(span_len):
        print(f'  {l:4d} bars : {span_len[l]:7d} spans')
    print('\nspans at least N bars long:')
    for thr in (4, 8, 12, 16, 24, 32, 48, 64):
        n = sum(c for l, c in span_len.items() if l >= thr)
        bars = sum(l * c for l, c in span_len.items() if l >= thr)
        note = ''
        if thr == 24:
            note = '   <- one full 384-subbeat training window'
        print(f'  >={thr:3d} : {n:7d} spans, {bars:9d} bars{note}')

    print('\nrule-following bars per key (the split the v1-v3 table needs):')
    for k in range(12):
        tag = ('pretrain-only' if k in PRETRAIN_ONLY else
               'UNSEEN' if k in UNSEEN else
               'both-seen' if k in BOTH_SEEN else 'finetune-only')
        print(f'  {ROOT_NAMES[k]:2s} (key {k:2d}) : {key_bars[k]:8d} bars, '
              f'{key_spans[k]:6d} spans   [{tag}]')
    thin = [k for k in UNSEEN if key_spans[k] < args.min_spans_per_key]
    if thin:
        print(f'  !! held-out keys {thin} have fewer than '
              f'{args.min_spans_per_key} spans: the unseen-key test cannot '
              'rest on them. Re-pick the held-out pair from the keys above, or '
              'transpose -- the LA tensor carries a pitch_shift_range per song '
              'and v0.42 was trained with that augmentation, so shifting a '
              'span into a thin key is in-distribution for the host.')

    if quality:
        print('\nchord qualities in the labels (the rule constrains none of them):')
        for q, c in quality.most_common(12):
            print(f'  {q:12s} {c}')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        cfg = {k: v for k, v in vars(args).items()
               if isinstance(v, (str, int, float, bool, type(None)))}
        with open(args.out, 'w') as f:
            json.dump({'config': cfg, 'offsets': OFFSETS,
                       'subbeats_per_bar': SUBBEATS_PER_BAR,
                       'songs': hits}, f)
        print(f'\nhit list -> {args.out}  ({len(hits)} songs)')


# ---------------------------------------------------------------------------
# ceiling
# ---------------------------------------------------------------------------

def cmd_ceiling(args):
    import torch

    with open(args.hits) as f:
        hits = json.load(f)
    songs = hits['songs']

    # idx -> relative path, as written by create_npy_dataset_from_midi and read
    # by preprocess_chords._load_txt. The basename is what MALD names its
    # labels after, which is the only link between the two.
    name_to_idx = {}
    with open(args.txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, rel = line.split('\t', 1)
            name_to_idx[os.path.splitext(os.path.basename(rel))[0]] = int(idx)

    matched = sorted(set(songs) & set(name_to_idx))
    print(f'{len(songs)} songs with spans, {len(name_to_idx)} in the CP index, '
          f'{len(matched)} in both')
    if not matched:
        sys.exit('no overlap -- the label basenames do not match the CP index, '
                 'so the two corpora are not keyed the same way')

    length = torch.load(args.cp[:-3] + '.length.pt', weights_only=True)
    start = torch.cumsum(length, dim=0) - length
    data = torch.load(args.cp, weights_only=True)
    poly = data.shape[1] // 4
    print(f'CP tensor {tuple(data.shape)}  ->  max_polyphony {poly}, '
          f'{SUBBEATS_PER_BAR} subbeats per bar')

    n_bars = empty = bass_empty = 0
    low_hit = bass_hit = any_hit = 0
    sat = 0
    for name in matched[:args.max_songs or None]:
        s = name_to_idx[name]
        if s >= len(length):
            continue
        base, n_sub = int(start[s]), int(length[s])
        for b0, L, key in songs[name]['spans']:
            for i in range(L):
                t = base + (b0 + i) * SUBBEATS_PER_BAR
                if t + 1 > base + n_sub:
                    break
                step = data[t].view(poly, 4)
                prog, pitch = step[:, 0].tolist(), step[:, 1].tolist()
                notes = [(p, g) for g, p in zip(prog, pitch)
                         if p != 255 and g != 254]
                n_bars += 1
                if len(notes) >= poly:
                    sat += 1
                if not notes:
                    empty += 1
                    bass_empty += 1
                    continue
                want = (key + OFFSETS[i % 4]) % 12
                if min(p for p, _ in notes) % 12 == want:
                    low_hit += 1
                if any(p % 12 == want for p, _ in notes):
                    any_hit += 1
                bass = [p for p, g in notes if g in BASS_PROGRAMS]
                if not bass:
                    bass_empty += 1
                elif min(bass) % 12 == want:
                    bass_hit += 1

    tot = max(n_bars, 1)
    print(f'\ndownbeats scored: {n_bars}')
    print(f'  no note at all on the downbeat      : {empty / tot:.3f}'
          '   (CP records onsets only, so a chord held over the barline leaves '
          'the downbeat step empty)')
    print(f'  polyphony cap reached at that step  : {sat / tot:.3f}'
          '   (notes beyond the cap were dropped at preprocessing)')
    print(f'  no bass-program note on the downbeat: {bass_empty / tot:.3f}')
    print('\nCEILING of each candidate v4 metric -- the rate at which the data '
          'itself satisfies it:')
    print(f'  lowest sounding pitch == rule root  : {low_hit / tot:.3f}')
    print(f'  lowest BASS-program pitch == root   : {bass_hit / tot:.3f}')
    print(f'  any sounding pitch == rule root     : {any_hit / tot:.3f}')
    print('\nv1-v3 scored a monophonic bass whose downbeat WAS the root, so '
          'their ceiling was 1.000. Whichever line above v4 scores, that number '
          'is the top of its table, and a model at the ceiling is following the '
          'rule as well as the data does.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('inspect', help='print the annotation format')
    q.add_argument('--root', required=True)
    q.add_argument('--n_examples', type=int, default=4)
    q.add_argument('--n_lines', type=int, default=12)
    q.set_defaults(func=cmd_inspect)

    c = sub.add_parser('census', help='count rule-following spans')
    c.add_argument('--chords', required=True, help='decoded_chords/ directory')
    c.add_argument('--ext', default='', help='only files with this extension')
    c.add_argument('--time_unit', default='beat', choices=['beat', 'second'],
                   help="unit of the label times. 'beat': a bar is "
                        '--beats_per_bar of them. Ignored for files with no times')
    c.add_argument('--sec_per_bar', type=float, default=None,
                   help='with --time_unit second and no downbeats, the fixed '
                        'bar length to assume. A guess: real tempo varies, so '
                        'prefer beats')
    c.add_argument('--beats_per_bar', type=int, default=BEATS_PER_BAR)
    c.add_argument('--bar_root', default='dominant',
                   choices=['dominant', 'single', 'first'])
    c.add_argument('--cover', type=float, default=0.99,
                   help='with --bar_root single, the fraction of the bar one '
                        'chord must cover')
    c.add_argument('--phase', type=int, default=0,
                   help='beats to drop before bar 0')
    c.add_argument('--phase_search', action='store_true',
                   help='try every bar phase per song and keep the best. MALD '
                        'does not verify downbeats, so the grid may be off by '
                        'beats; this fits the offset instead of trusting it, '
                        'and the printed phase spread says how much that '
                        'mattered')
    c.add_argument('--min_span', type=int, default=4,
                   help='shortest span to count, in bars (4 = one I-IV-V-I)')
    c.add_argument('--min_spans_per_key', type=int, default=200)
    c.add_argument('--max_songs', type=int, default=0)
    c.add_argument('--out', default='')
    c.set_defaults(func=cmd_census)

    e = sub.add_parser('ceiling', help='metric ceiling on the found spans')
    e.add_argument('--hits', required=True, help='census --out JSON')
    e.add_argument('--cp', required=True, help='data/la_cp16_v2.pt')
    e.add_argument('--txt', required=True, help='data/la_cp16_v2.txt')
    e.add_argument('--max_songs', type=int, default=2000)
    e.set_defaults(func=cmd_ceiling)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
