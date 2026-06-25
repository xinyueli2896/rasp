"""
Continuation inference for CPYinyangTransformer (rule-conditioned CP transformer).

Mirrors cp_transformer_inference.py but uses the adapter model.

Usage
-----
  python -m midi_adapter.infer_cp_yinyang \\
      --model_path checkpoints/.../last.ckpt \\
      --midi_path  input/my_prompt.mid \\
      --approach chord --encoder_injected --n_skip 1 \\
      --prompt_length 4 --generation_length 32 --n_samples 2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_transformer import RoFormerSymbolicTransformer, CPTokenizer
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from midi_adapter.cp_yinyang import CPYinyangTransformer


# ---------------------------------------------------------------------------
# decode_output  (matches cp_transformer_inference.py exactly)
# ---------------------------------------------------------------------------

def decode_output(outputs, save_path=None, tempo=120.0, ratio=1.0, velocity=100,
                  with_velocity=False, extra_instruments=None, fixed_program=None):
    tokenizer = CPTokenizer(with_velocity=with_velocity)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    if not isinstance(ratio, tuple):
        ratio = (ratio,) * len(outputs)
    for r, output in zip(ratio, outputs):
        instrument_map = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            start_time = time_step * time_step_length
            for i in range(0, len(content), 2):
                a_token = int(content[i].item())
                if a_token == tokenizer.eos_token:
                    break
                if i + 1 >= len(content):
                    print('Incomplete note @', time_step, i)
                    break
                b_token = int(content[i + 1].item())
                if with_velocity:
                    program      = a_token % 128
                    velocity_bin = a_token // 128
                    note_velocity = velocity_bin * 8 + 4
                    pitch_duration = b_token - 16 * 128
                else:
                    program        = a_token
                    note_velocity  = velocity
                    pitch_duration = b_token - 128
                pitch    = pitch_duration % 128
                duration = pitch_duration // 128
                if program < 0 or program >= 128:
                    print('Invalid program:', program, '@', time_step, i)
                    break
                if with_velocity and (velocity_bin < 0 or velocity_bin >= 16):
                    print('Invalid velocity bin:', velocity_bin, '@', time_step, i)
                    break
                if pitch < 0 or pitch >= 128:
                    print('Invalid pitch:', pitch, '@', time_step, i)
                    break
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print('Invalid duration:', duration, '@', time_step, i)
                    break
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                if program not in instrument_map:
                    if program == 127:
                        instrument_map[program] = pretty_midi.Instrument(0, is_drum=True)
                    else:
                        instrument_map[program] = pretty_midi.Instrument(
                            fixed_program if fixed_program is not None else program)
                    midi.instruments.append(instrument_map[program])
                instrument_map[program].notes.append(
                    pretty_midi.Note(velocity=note_velocity, pitch=pitch,
                                     start=start_time * r, end=end_time * r))
    if extra_instruments is not None:
        for instrument in extra_instruments:
            midi.instruments.append(instrument)
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        midi.write(save_path)
    return midi


# ---------------------------------------------------------------------------
# decompress  (mirrors cp_transformer_inference.py)
# ---------------------------------------------------------------------------

def decompress(model: CPYinyangTransformer, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0).cuda()
    return model.base.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda())


# ---------------------------------------------------------------------------
# continuation  (mirrors cp_transformer_inference.py)
# ---------------------------------------------------------------------------

def continuation(model: CPYinyangTransformer, midi_path: str,
                 prompt_length: int = 4, generation_length: int = 32,
                 temperature: float = 1.0, n_samples: int = 1):
    x = decompress(model, preprocess_midi(midi_path, 4)[0])
    print(x.shape)
    x = x[:, :prompt_length]

    out_base = os.path.splitext(os.path.basename(midi_path))[0]
    decode_output([x[:, i, :] for i in range(x.shape[1])],
                  f'output/{out_base}_prompt.mid')

    with torch.no_grad():
        x = x.repeat(n_samples, 1, 1)
        output = model.global_sampling(
            x, temperature=temperature,
            max_seq_len=prompt_length + generation_length,
        )

    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        decode_output(output_i,
                      f'output/{out_base}_temp{temperature}_continuation_{i}.mid')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, approach: str, encoder_injected: bool,
               encoder_type: str, rule_mode: str,
               model_size: int, adapter_rank: int, n_skip: int,
               base_ckpt: str | None = None) -> CPYinyangTransformer:
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    base   = RoFormerSymbolicTransformer(size=model_size, max_lr=max_lr, with_velocity=False)
    model  = CPYinyangTransformer(
        base,
        adapter_rank     = adapter_rank,
        n_skip           = n_skip,
        encoder_injected = encoder_injected,
        encoder_type     = encoder_type,
        rule_mode        = rule_mode,
        approach         = approach,
    )

    raw = torch.load(model_path, map_location='cpu')
    if 'state_dict' in raw:
        full_state = {k[len('model.'):]: v for k, v in raw['state_dict'].items()
                      if k.startswith('model.')}
        model.load_state_dict(full_state, strict=False)
    else:
        if base_ckpt and os.path.exists(base_ckpt):
            bstate = torch.load(base_ckpt, map_location='cpu')
            if 'state_dict' in bstate:
                bstate = bstate['state_dict']
            base.load_state_dict(bstate)
        model.load_state_dict(raw, strict=False)

    model.cuda()
    model.eval()
    model.save_name = os.path.basename(model_path)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',       type=str, required=True)
    p.add_argument('--base_ckpt',        type=str, default=None)
    p.add_argument('--midi_path',        type=str, required=True)
    p.add_argument('--prompt_length',    type=int, default=4)
    p.add_argument('--generation_length',type=int, default=32)
    p.add_argument('--temperature',      type=float, default=1.0)
    p.add_argument('--n_samples',        type=int, default=1)
    p.add_argument('--approach',         type=str, default='chord', choices=['bass', 'chord'])
    p.add_argument('--encoder_injected', action='store_true')
    p.add_argument('--encoder_type',     type=str, default='embedding', choices=['embedding', 'token_mlp'])
    p.add_argument('--rule_mode',        type=str, default='current')
    p.add_argument('--model_size',       type=int, default=1)
    p.add_argument('--adapter_rank',     type=int, default=256)
    p.add_argument('--n_skip',           type=int, default=1)
    args = p.parse_args()

    model = load_model(args.model_path, args.approach, args.encoder_injected,
                       args.encoder_type, args.rule_mode,
                       args.model_size, args.adapter_rank, args.n_skip,
                       args.base_ckpt)

    continuation(model, args.midi_path,
                 prompt_length    = args.prompt_length,
                 generation_length= args.generation_length,
                 temperature      = args.temperature,
                 n_samples        = args.n_samples)
