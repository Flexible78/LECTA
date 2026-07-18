import torch
import os
import numpy as np
import torch
import torchaudio
from f5_tts.model import DiT,CFM
from f5_tts.infer.utils_infer import load_checkpoint
from typing import Tuple

def get_tokenizer(dataset_name='models/vocab.txt'):
    with open(dataset_name, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

def load_models(model_cls=DiT, ckpt_path=None, device='cpu'):
    vocab_char_map, vocab_size = get_tokenizer()
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(
            method="euler",
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True)

    return model

speakers_data = {}

def load_speakers_archive(archive_file='models/speakers_data.pt', device=None):
    global speakers_data
    if not os.path.exists(archive_file):
        raise FileNotFoundError(f"Archive file {archive_file} not found")
    
    data = torch.load(archive_file, map_location=device)

    if device is not None:
        for speaker_id in data:
            for key, value in data[speaker_id].items():
                if torch.is_tensor(value):
                    data[speaker_id][key] = value.to(device)
    
    speakers_data = data
    spk_list = []
    for i in speakers_data:
        spk_list.append((speakers_data[i]['name'],i))
    return spk_list

def generate_audio(
    gen_text: str,
    model_obj,
    vocoder,
    ref_data,
    nfe_step: int = 10,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 0.9,
    target_rms: float = 0.1
) -> Tuple[np.ndarray, int]:
    
    r_audio = ref_data['audio']
    rms = ref_data['rms']
    ref_text = ref_data['text']
    ref_audio_len = ref_data['audio_len']

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    text_list = [ref_text + gen_text]
    final_text_list = [list(text_list[0])]

    duration = ref_audio_len + int(ref_audio_len / ref_data['text_len'] * len(gen_text.encode("utf-8")) / speed)

    with torch.inference_mode():
        generated, _ = model_obj.sample(
            cond=r_audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        
        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated = generated.permute(0, 2, 1)
        
        if hasattr(vocoder, 'decode'):
            generated_wave = vocoder.decode(generated)
        else:
            generated_wave = vocoder(generated)
        
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms
        
        generated_wave = generated_wave.squeeze().cpu().numpy()
    
    return generated_wave, 24000

def prep_audio_from_gradio(audio_input, target_sample_rate=24000, device=None):

    if isinstance(audio_input, tuple):
        sr, audio_array = audio_input
        if audio_array.dtype == np.int16:
            audio_array = audio_array / 32768.0
        audio = torch.from_numpy(audio_array).float().unsqueeze(0)  # (1, T)
    else:
        raise ValueError(f"Ожидался tuple (sr, np.ndarray), получено: {type(audio_input)}")

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    else:
        audio = audio

    target_rms_val = 0.1
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms_val:
        audio = audio * target_rms_val / rms
        rms = torch.tensor(target_rms_val)

    audio_len = audio.shape[-1] // 256

    return {
        'audio': audio.to(device),
        'rms': rms,
        'audio_len': audio_len
    }

def get_speaker_data(speaker_id=None):
    speaker_key = str(speaker_id)
    if speaker_key not in speakers_data:
        raise ValueError(f"Speaker {speaker_id} not found in archive")
    data = speakers_data[speaker_key].copy()

    return data
