import torch
import numpy as np
import torchaudio
from pathlib import Path
from libs.utils import download_model, models_path
from vocos import Vocos
from .model import DiT, CFM
from typing import Tuple

class F5Model:
    def __init__(self):
        self.vocab = None
        self.speakers_data = None
        self.speakers = None
        self.model = None
        # Уважает глобальный переключатель GPU/CPU из tts/__init__.py
        try:
            from libs.tts import get_device
            self.device = get_device()
        except ImportError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_tokenizer(self, dataset_name=models_path/'vocab.txt'):
        vocab_path = Path(dataset_name)
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

        self.vocab = (vocab_char_map, vocab_size)

    def load(self, model_cls=DiT, ckpt_path=None, model_ver=None):
        self.get_tokenizer()
        if model_ver == 5:
            model_file = 'F5TTS_v1_Base_v4_winter/model_212000.safetensors'
            hf_url = f"https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/{model_file}"
            ckpt_path = models_path / Path(model_file)
        else:
            model_file = 'espeech_tts_rlv2.pt'
            hf_url = f"https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2/resolve/main/{model_file}"
            ckpt_path = models_path / "ESpeech-TTS-1_RL-V2" / model_file
        
        m, status = download_model(hf_url, ckpt_path)
        if m is None:
            return status
        if self.speakers is None:
            self.speakers_list(device=self.device)
        vocab_char_map, vocab_size = self.vocab
        
        # --- ПРАВКА: Строго 512 для всех моделей, чтобы не было size mismatch ---
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
        ).to(self.device)

        self.model = load_checkpoint(model, str(ckpt_path) if isinstance(ckpt_path, Path) else ckpt_path, self.device, dtype=None, use_ema=True)

        return model

    def speakers_list(self, archive_file=models_path/'speakers_data.pt', device=None):
        archive_path = Path(archive_file)
        self.speakers = [] 

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive file {archive_path} not found")
        
        if device is not None:
            data = torch.load(str(archive_path), map_location=device)
        else:
            data = torch.load(str(archive_path))

        if device is not None:
            for speaker_id in data:
                for key, value in data[speaker_id].items():
                    if torch.is_tensor(value):
                        data[speaker_id][key] = value.to(device)
        
        self.speakers_data = data
        for i in sorted(data, key=int):
            self.speakers.append((data[i]['name'], i))
        return self.speakers

class F5Synth:
    def __init__(self, model):
        self.model = model
        self.vocoder = None

    def generate_audio(
        self,
        gen_text: str,
        ref_data,
        nfe_step: int = 16,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 0.9,
        target_rms: float = 0.1
    ) -> Tuple[np.ndarray, int]:
        
        if len(gen_text) <= 40:
            gen_text = f'{gen_text}" "'
        r_audio = ref_data['audio']
        rms = ref_data['rms']
        ref_text = ref_data.get('text', "")
        ref_audio_len = ref_data['audio_len']

        # --- ПРАВКА: Защита от пустого текста образца (чинит IndexError) ---
        if not ref_text:
            ref_text = " "
        elif len(ref_text) > 0 and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "

        text_list = [ref_text + gen_text]
        final_text_list = [list(text_list[0])]

        text_len_safe = max(1, ref_data.get('text_len', 1))
        duration = ref_audio_len + int(ref_audio_len / text_len_safe * len(gen_text.encode("utf-8")) / speed)

        with torch.inference_mode():
            generated, _ = self.model.model.sample(
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
            
            if hasattr(self.vocoder, 'decode'):
                generated_wave = self.vocoder.decode(generated)
            else:
                generated_wave = self.vocoder(generated)
            
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms
            
            generated_wave = generated_wave.squeeze().cpu().numpy()
        
        return generated_wave, 24000

    def load_vocoder(self):
        model_path = models_path / 'vocos-mel-24khz'
        model_path.mkdir(exist_ok=True)
        try:
            model_file = 'pytorch_model.bin'
            config_file = 'config.yaml'
            model_filepath = model_path / model_file
            config_filepath = model_path / config_file
            hf_url = f"https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/{model_file}"
            m, status = download_model(hf_url, model_filepath)
            if m is None:
                return status
            hf_url = f"https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/{config_file}"
            m, status = download_model(hf_url, config_filepath)
            if m is None:
                return status
            vocoder = Vocos.from_hparams(str(config_filepath))
            state_dict = torch.load(model_filepath, map_location="cpu")
            vocoder.load_state_dict(state_dict)
            self.vocoder = vocoder.eval().to(self.model.device)
            return "Вокодер загружен"
        except Exception as e:
            return(f"Ошибка: {type(e).__name__}: {e}")

    def synth_audio(self, text, speaker_id=None, speed=1.0, noise=16, ref_audio=None, ref_text=''):
        if self.vocoder is None:
            self.load_vocoder()

        data = {}
        if ref_audio is None:
            data=self.model.speakers_data[speaker_id].copy()
        else:
            data = prep_audio_from_gradio(ref_audio, device=self.model.device)
            data['text'] = ref_text
            data['text_len'] = len(ref_text.encode("utf-8"))

        audio_wave, sample_rate = self.generate_audio(
            f' {text} ',
            speed=speed,
            nfe_step=noise,
            ref_data=data
        )
        audio_wave = (audio_wave * 32767).astype(np.int16)
        return audio_wave, sample_rate

def prep_audio(
    audio_input, 
    target_sample_rate: int = 24000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    sample_rate, audio_numpy = audio_input
    audio_numpy = audio_numpy / 32768.0
    audio_tensor = torch.from_numpy(audio_numpy).float()
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.T
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    else:
        audio_tensor = audio_tensor.unsqueeze(0)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        audio_tensor = resampler(audio_tensor)

    target_rms = 0.1
    rms = torch.sqrt(torch.mean(torch.square(audio_tensor)))
    audio_tensor = audio_tensor * target_rms / rms

    return audio_tensor.to(device), rms.to(device), target_sample_rate

def prep_audio_from_gradio(audio_input, target_sample_rate=24000, device=None):

    if isinstance(audio_input, tuple):
        sr, audio_array = audio_input
        if audio_array.dtype == np.int16:
            audio_array = audio_array / 32768.0
        audio = torch.from_numpy(audio_array).float().unsqueeze(0)
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

def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)