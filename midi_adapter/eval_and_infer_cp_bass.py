"""
Combined evaluation + inference script for the CP bass transformer.

For each key group (pretrain_keys, finetune_new, unseen):
  1. Teacher-forced accuracy on N_SONGS synthetic songs
  2. One autoregressive generation example
  3. ASCII piano-roll + root-tracking printout
  4. Save generated MIDI to --out_dir

Summary accuracy table printed at the end (mirrors evaluate.py from the RASP repo).

Usage
-----
  # Base model only
  python -m midi_adapter.eval_and_infer_cp_bass \\
      --base_ckpt checkpoints/cp_bass_size1_pretrain.pt

  # With adapter
  python -m midi_adapter.eval_and_infer_cp_bass \\
      --base_ckpt    checkpoints/cp_bass_size1_pretrain.pt \\
      --adapter_ckpt checkpoints/cp_yinyang_size1_rank256.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer
from midi_adapter.cp_yinyang import CPYinyangTransformer
from midi_adapter.chord_tokenizer import N_QUALITIES, NO_CHORD_TOKEN
from midi_adapter.generate_synthetic_bass import (
    generate_song, _preprocess_pm,
    SUBBEATS_PER_BAR, CONSTANT_TEMPO, DURATION_TEMPLATES, OFFSETS,
)

try:
    import pretty_midi
except ImportError:
    print('pretty_midi is required: pip install pretty_midi')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
TRAIN_LENGTH = 384

KEY_GROUPS = {
    'pretrain_keys': [0, 2, 4, 5, 7, 9, 11],
    'finetune_new':  [1, 3, 10],
    'unseen':        [6, 8],
}


# ---------------------------------------------------------------------------
# Batch generation (teacher-forced evaluation)
# ---------------------------------------------------------------------------

def make_batch(keys, n_songs, n_bars, device):
    from midi_adapter.chord_tokenizer import chord_str_to_token

    n_subbeats = n_bars * SUBBEATS_PER_BAR
    x_list, ct_list = [], []

    for i in range(n_songs):
        key = keys[i % len(keys)]
        pm, xf_chords = generate_song(n_bars=n_bars, key=key)
        data, _ = _preprocess_pm(pm, n_subbeats)
        x_list.append(data)

        bar_tokens = []
        for b in range(n_bars):
            root = (key + OFFSETS[b % 4]) % 12
            chord_str = xf_chords[b][1]
            tok = chord_str_to_token(chord_str)
            bar_tokens.append(tok)
        ct_list.append(torch.tensor(bar_tokens, dtype=torch.long))

    x            = torch.stack(x_list).to(device)
    chord_tokens = torch.stack(ct_list).to(device)
    pitch_shift  = torch.zeros(n_songs, dtype=torch.long, device=device)
    return x, chord_tokens, pitch_shift


# ---------------------------------------------------------------------------
# Teacher-forced accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(model, x, chord_tokens, pitch_shift, base, window_size=TRAIN_LENGTH):
    B, T, _ = x.shape
    n_win_bars = window_size // SUBBEATS_PER_BAR
    correct = total = 0

    for start_sub in range(0, T - window_size + 1, SUBBEATS_PER_BAR):
        start_bar = start_sub // SUBBEATS_PER_BAR
        end_bar   = start_bar + n_win_bars

        x_win  = x[:, start_sub:start_sub + window_size, :]
        ct_win = chord_tokens[:, start_bar:end_bar]

        x_proc = base.preprocess(x_win, pitch_shift)

        if isinstance(model, CPYinyangTransformer):
            logits = model(x_proc, ct_win)
        else:
            logits = base(x_proc)

        logits = logits.view(B, window_size, 8, -1)
        pitch_logits = logits[:, ::16, 1, :]
        pred_pitch   = pitch_logits.argmax(-1) % 128

        expected_root  = ct_win // N_QUALITIES
        expected_pitch = 36 + expected_root

        valid    = ct_win != NO_CHORD_TOKEN
        correct += (pred_pitch == expected_pitch)[valid].sum().item()
        total   += valid.sum().item()

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Autoregressive generation helpers (no KV cache — RoFormerEncoder unsupported)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _local_sample(base, h, temperature, max_len=32):
    B = h.shape[0]
    tok = base.tokenizer
    h_emb = h.unsqueeze(1)
    generated = []
    done = torch.zeros(B, dtype=torch.bool, device=h.device)

    for _ in range(max_len):
        seq = torch.cat(
            [h_emb] + [base.local_embedding(t.unsqueeze(1)) for t in generated],
            dim=1,
        )
        out    = base.local_decoder(seq, attention_mask=base.buffered_future_mask(seq))[0]
        logits = base.final_decoder(out[:, -1])

        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)

        nxt[done] = tok.pad_token
        done = done | (nxt.squeeze(1) == tok.eos_token)
        generated.append(nxt.squeeze(1))
        if done.all():
            break

    if not generated:
        return torch.full((B, 1), tok.pad_token, dtype=torch.long, device=h.device)
    return torch.stack(generated, dim=1)


@torch.no_grad()
def ar_generate(base, key, n_prompt_bars, n_gen_bars, temperature, device):
    """
    Generate n_gen_bars bars autoregressively, conditioned on a synthetic prompt.

    Returns
    -------
    all_tokens   : list[(1, subseq_len)] covering prompt + generated subbeats
    n_prompt_sub : int  — number of subbeats that are prompt (for split marking)
    """
    n_total = (n_prompt_bars + n_gen_bars) * SUBBEATS_PER_BAR

    if n_prompt_bars > 0:
        pm, _ = generate_song(n_bars=n_prompt_bars, key=key)
        data, _ = _preprocess_pm(pm, n_prompt_bars * SUBBEATS_PER_BAR)
        data = data.unsqueeze(0).to(device)
        ps   = torch.zeros(1, dtype=torch.long, device=device)
        prompt = base.preprocess(data, ps)                      # (1, T_p, 8)
    else:
        prompt = torch.zeros(1, 1, 8, dtype=torch.long, device=device)

    T_p = prompt.shape[1]
    sos = base.global_sos.view(1, 1, -1)
    h_enc, _ = base.local_encode(prompt)
    h_hist = torch.cat([sos, h_enc.view(1, T_p, -1)], dim=1)

    # Keep prompt tokens so the MIDI starts from bar 0 (tonic)
    prompt_tokens = [prompt[:, t, :] for t in range(T_p)]

    gen_tokens = []
    for step in range(T_p, n_total):
        h_out  = base.model(h_hist, attention_mask=base.buffered_future_mask(h_hist))[0]
        h_last = h_out[:, -1, :]
        y_next = _local_sample(base, h_last, temperature)
        gen_tokens.append(y_next)
        h_enc2, _ = base.local_encode(y_next.unsqueeze(1))
        h_hist = torch.cat([h_hist, h_enc2.unsqueeze(1)], dim=1)

    return prompt_tokens + gen_tokens, T_p


# ---------------------------------------------------------------------------
# Decode tokens → PrettyMIDI  (matches original repo decode_output)
# ---------------------------------------------------------------------------

def decode_to_midi(sampled, tokenizer, tempo=CONSTANT_TEMPO, velocity=100,
                   fixed_program=None):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step = 60.0 / tempo / 4
    inst_map: dict[int, pretty_midi.Instrument] = {}

    for t, tok_batch in enumerate(sampled):
        content = tok_batch[0]
        start   = t * time_step

        for i in range(0, len(content), 2):
            prog = int(content[i].item())
            if prog == tokenizer.eos_token:
                break
            if i + 1 >= len(content):
                break

            pd = int(content[i + 1].item()) - 128
            pitch    = pd % 128
            duration = pd // 128

            if prog < 0 or prog >= 128:
                break
            if not (0 <= pitch < 128):
                break
            if not (0 <= duration < len(DURATION_TEMPLATES)):
                break

            end = start + DURATION_TEMPLATES[duration] * time_step

            if prog not in inst_map:
                p = fixed_program if fixed_program is not None else prog
                if prog == 127:
                    inst_map[prog] = pretty_midi.Instrument(0, is_drum=True)
                else:
                    inst_map[prog] = pretty_midi.Instrument(program=p)
                pm.instruments.append(inst_map[prog])

            inst_map[prog].notes.append(
                pretty_midi.Note(velocity=velocity, pitch=pitch,
                                 start=start, end=end)
            )
    return pm


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

PITCH_RANGE = (24, 72)   # C1–C5: typical bass register

def _pitch_name(midi_pitch):
    return ROOT_NAMES[midi_pitch % 12] + str(midi_pitch // 12 - 1)


def print_piano_roll(pm, n_bars, title=''):
    """
    ASCII piano roll: rows = pitches (PITCH_RANGE), columns = bars.
    Each bar column is one char: the note-name of the first note in that bar,
    or '.' if silent.
    """
    time_step = 60.0 / CONSTANT_TEMPO / 4   # seconds per subbeat
    bar_duration = SUBBEATS_PER_BAR * time_step

    lo, hi = PITCH_RANGE
    n_show = min(n_bars, 16)    # cap display width

    # Build grid: pitch × bar → note char or None
    grid = [[None] * n_show for _ in range(hi - lo + 1)]
    for inst in pm.instruments:
        for note in inst.notes:
            bar = int(note.start / bar_duration)
            if 0 <= bar < n_show and lo <= note.pitch <= hi:
                row = hi - note.pitch
                if grid[row][bar] is None:
                    grid[row][bar] = ROOT_NAMES[note.pitch % 12]

    if title:
        print(f'  Piano roll — {title}')

    # Bar header (prefix must match row prefix width = 2 + 6 + 2 = 10)
    bar_hdr = '  ' + ' ' * 8 + ''.join(f'{b:<4}' for b in range(n_show))
    print(bar_hdr)
    print('  ' + '-' * (8 + n_show * 4))

    # Print every 3rd pitch row to keep it compact (12 rows across C1–C5)
    for row_idx, row in enumerate(grid):
        pitch = hi - row_idx
        if pitch % 12 != 0:   # only print C rows for a skeleton
            continue
        name = _pitch_name(pitch)
        cells = ''.join(
            f'{(c or "."):.<4}' for c in row
        )
        print(f'  {name:<6}| {cells}')

    print()


def print_root_table(all_tokens, key, n_prompt_sub=0, n_bars_show=16, title=''):
    """
    Show bar-by-bar expected root vs. the note played at the bar-start subbeat.

    all_tokens   : list[(1, subseq_len)] covering (prompt + generated) subbeats
    n_prompt_sub : number of prompt subbeats (marked with '[' ']' in output)
    """
    tokenizer_eos  = 3329   # CPTokenizer eos_token value (n_normal_tokens + 1)

    n_bars = min(len(all_tokens) // SUBBEATS_PER_BAR, n_bars_show)
    n_prompt_bars = n_prompt_sub // SUBBEATS_PER_BAR

    expected_roots = [(key + OFFSETS[b % 4]) % 12 for b in range(n_bars)]

    # Extract pitch at every bar-start subbeat (index 0, 16, 32, …)
    gen_pitches = []
    for b in range(n_bars):
        tok_batch = all_tokens[b * SUBBEATS_PER_BAR]
        content   = tok_batch[0].tolist()
        pitch     = None
        for i in range(0, len(content) - 1, 2):
            prog = content[i]
            if prog == tokenizer_eos or prog >= 128:
                break
            pd = content[i + 1] - 128
            p  = pd % 128
            d  = pd // 128
            if 0 <= p < 128 and 0 <= d < len(DURATION_TEMPLATES):
                pitch = p
                break
        gen_pitches.append(pitch)

    # ---- print ----
    if title:
        print(f'  {title}')

    # Mark prompt bars with brackets
    def _bar_label(b):
        s = str(b)
        return f'[{s}]' if b < n_prompt_bars else s

    bar_line   = '  Bar   : ' + ''.join(f'{_bar_label(b):<5}' for b in range(n_bars))
    exp_line   = '  Expect: ' + ''.join(f'{ROOT_NAMES[r]:<5}' for r in expected_roots)

    gen_names  = []
    match_syms = []
    for b, (exp, gp) in enumerate(zip(expected_roots, gen_pitches)):
        if gp is None:
            gn, sym = '--', ' '
        else:
            gn  = ROOT_NAMES[gp % 12]
            sym = '»' if b < n_prompt_bars else ('✓' if (gp % 12) == exp else '✗')
        gen_names.append(f'{gn:<5}')
        match_syms.append(f'{sym:<5}')

    gen_line   = '  Gen   : ' + ''.join(gen_names)
    match_line = '  Match : ' + ''.join(match_syms)

    # Only score generated (non-prompt) bars
    n_correct = sum(
        1 for b, (exp, gp) in enumerate(zip(expected_roots, gen_pitches))
        if b >= n_prompt_bars and gp is not None and (gp % 12) == exp
    )
    n_valid = sum(
        1 for b, gp in enumerate(gen_pitches)
        if b >= n_prompt_bars and gp is not None
    )
    pct = n_correct / max(n_valid, 1) * 100

    print(bar_line)
    print(exp_line)
    print(gen_line)
    print(match_line)
    print(f'  Root accuracy (AR, generated bars only): {n_correct}/{n_valid} = {pct:.1f}%')
    print(f'  [brackets] = prompt bars  (» = given, not scored)')
    print()


# ---------------------------------------------------------------------------
# Main evaluation + inference loop
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- load base model ----
    max_lr = 5e-5 if args.model_size >= 2 else 1e-4
    base   = RoFormerSymbolicTransformer(
        size=args.model_size, max_lr=max_lr, with_velocity=False,
    )
    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        base.load_state_dict(state)
        print(f'Loaded base: {args.base_ckpt}')
    else:
        print('WARNING: no base checkpoint — using random weights.')

    # ---- optionally load adapter ----
    use_adapter = args.adapter_ckpt and os.path.exists(args.adapter_ckpt)
    if use_adapter:
        model = CPYinyangTransformer(base, adapter_rank=args.adapter_rank,
                                     n_skip=args.n_skip)
        state = torch.load(args.adapter_ckpt, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        print(f'Loaded adapter: {args.adapter_ckpt}')
    else:
        model = base
        print('Evaluating base CP transformer (no adapter).')

    model = model.to(device).eval()
    base  = base.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- per-group loop ----
    results = {}

    for group_name, keys in KEY_GROUPS.items():
        key_str = ', '.join(ROOT_NAMES[k] for k in keys)
        print()
        print('=' * 72)
        print(f'  KEY GROUP : {group_name}  ({key_str})')
        print('=' * 72)

        # 1. Teacher-forced accuracy
        x, chord_tokens, pitch_shift = make_batch(
            keys, n_songs=args.n_songs, n_bars=args.n_bars, device=device,
        )
        acc = compute_accuracy(model, x, chord_tokens, pitch_shift, base)
        results[group_name] = acc
        print(f'  Teacher-forced accuracy: {acc:.4f}')
        print()

        # 2. Autoregressive generation (one example, one key from group)
        demo_key = keys[0]
        key_name = ROOT_NAMES[demo_key]
        print(f'  Generating {args.n_gen_bars} bars (key={key_name}, '
              f'prompt={args.n_prompt_bars} bars, T={args.temperature}) ...')

        all_tokens, n_prompt_sub = ar_generate(
            base, demo_key,
            n_prompt_bars=args.n_prompt_bars,
            n_gen_bars=args.n_gen_bars,
            temperature=args.temperature,
            device=device,
        )
        n_total_bars = args.n_prompt_bars + args.n_gen_bars

        # 3. Root tracking table (prompt bars shown with brackets, not scored)
        print_root_table(
            all_tokens, demo_key,
            n_prompt_sub=n_prompt_sub,
            n_bars_show=n_total_bars,
            title=f'Root tracking (key={key_name}, group={group_name})',
        )

        # 4. Decode + save MIDI (includes prompt so file starts on tonic)
        pm = decode_to_midi(all_tokens, base.tokenizer,
                            fixed_program=args.fixed_program)
        midi_path = os.path.join(args.out_dir, f'{group_name}_key{demo_key}.mid')
        pm.write(midi_path)
        print(f'  Saved MIDI → {midi_path}')

        # 5. ASCII piano roll
        print_piano_roll(pm, n_total_bars,
                         title=f'{group_name} / key={key_name} (bar 0 = prompt)')

    # ---- summary table ----
    print()
    print('=' * 72)
    print('SUMMARY — Teacher-forced chord-root accuracy')
    print('=' * 72)
    col_w = 18
    print(f'{"Key group":<{col_w}}  {"Keys":<28}  {"n_songs":>8}  {"accuracy":>10}')
    print('-' * 68)
    for group_name, keys in KEY_GROUPS.items():
        acc = results[group_name]
        print(f'{group_name:<{col_w}}  {str(keys):<28}  {args.n_songs:>8}  {acc:>10.4f}')
    print()

    pretrain_acc = results['pretrain_keys']
    unseen_acc   = results['unseen']
    print(f'  unseen / pretrain ratio: {unseen_acc / max(pretrain_acc, 1e-6):.4f}')
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate + infer CP bass transformer per key group'
    )
    p.add_argument('--base_ckpt',     type=str, required=True)
    p.add_argument('--adapter_ckpt',  type=str, default=None)
    p.add_argument('--model_size',    type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--adapter_rank',  type=int, default=256)
    p.add_argument('--n_skip',        type=int, default=4)
    p.add_argument('--n_songs',       type=int, default=20,
                   help='Songs per key group for teacher-forced accuracy')
    p.add_argument('--n_bars',        type=int, default=32,
                   help='Bars per song for accuracy eval (>= 24)')
    p.add_argument('--n_gen_bars',    type=int, default=16,
                   help='Bars to generate for the AR demo')
    p.add_argument('--n_prompt_bars', type=int, default=1,
                   help='Prompt bars before free generation (1 = just the tonic bar, '
                        'analogous to RASP prompt_len=1)')
    p.add_argument('--temperature',   type=float, default=1.0)
    p.add_argument('--fixed_program', type=int, default=32,
                   help='MIDI program for all notes (32 = acoustic bass)')
    p.add_argument('--out_dir',       type=str, default='output/midi_eval')
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)
