import numpy as np
import time
import re
import logging
import json
import onnxruntime
import requests
from zipfile import ZipFile
from pathlib import Path
from tokenizers.implementations import BertWordPieceTokenizer
from libs.utils import download_model, models_path
from .g2p import G2P

class Model:
    def __init__(self, model_name=None):
        model_path = self.get_model_by_name(model_name)
        onnx_providers = onnxruntime.get_available_providers()
        providers = [p for p in onnx_providers if p in ["CUDAExecutionProvider", "CPUExecutionProvider"]]

        sess_options = onnxruntime.SessionOptions()        
        logging.info(f"Loading model from {model_path}")
        
        model_onnx_path = model_path / "model.onnx"
        self.onnx = onnxruntime.InferenceSession(str(model_onnx_path), sess_options=sess_options, providers=providers)

        self.dic = {}
        probs = {}
        dictionary_path = model_path / "dictionary"
        with open(dictionary_path, encoding='utf-8') as f:
            for line in f:
                items = line.split(maxsplit=2)
                prob = float(items[1])
                if probs.get(items[0], 0) < prob:
                    self.dic[items[0]] = items[2]
                    probs[items[0]] = prob

        config_path = model_path / "config.json"
        self.config = json.loads(config_path.read_text(encoding='utf-8'))
        self.speakers = self.speakers_list()

        self.g2p = G2P(self.dic, self.config["phoneme_id_map"])
        bert_vocab_path = model_path / "bert" / "vocab.txt"
        if bert_vocab_path.exists():
            self.tokenizer = BertWordPieceTokenizer(vocab=str(bert_vocab_path), unk_token="[UNK]", lowercase=True)
            bert_onnx_path = model_path / "bert" / "model.onnx"
            self.bert_onnx = onnxruntime.InferenceSession(str(bert_onnx_path), sess_options=sess_options, providers=providers)
        else:
            self.tokenizer = None

    def get_model_by_name(self, model_name) -> Path:
        models_path.mkdir(exist_ok=True)
        model_dir = models_path / f"vosk-model-tts-ru-{model_name}-multi"
        
        if not model_dir.exists():
            print(f'Загрузка Vosk TTS {model_name}')
            url = f"https://myfreenet.ru/models/vosk-model-tts-ru-{model_name}-multi.zip"
            zip_path = models_path / model_dir.with_suffix('.zip')
            m, status = download_model(url, zip_path)
            if m is not None:
                with ZipFile(zip_path, "r") as model_ref:
                    model_ref.extractall(str(models_path))
            zip_path.unlink(missing_ok=True)
        
        return model_dir

    def speakers_list(self):
        if self.config["speaker_id_map"]:
            return [
                (i, speaker)
                for i, speaker in self.config["speaker_id_map"].items()
            ]  
        else:
            return []

class Synth:

    def __init__(self, model):
        self.model = model

    def audio_float_to_int16(self,
        audio: np.ndarray, max_wav_value: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16 range"""
        audio_norm = audio * max_wav_value
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    def get_word_bert(self, text, nopunc=False):
        tokens = self.model.tokenizer.encode(text.replace("+", "").replace("_", ""))
        bert = self.model.bert_onnx.run(
            None,
            {
               "input_ids": [tokens.ids],
               "attention_mask": [tokens.attention_mask],
               "token_type_ids": [tokens.type_ids],
            }
        )[0]

        pattern = "[-,.?!;:\"]"
        selected = []
        for i, t in enumerate(tokens.tokens):
            if t[0] != '#':
                if not (nopunc and re.match(pattern, t)):
                    selected.append(i)
        bert = bert[selected]
        return bert

    def synth_audio(self, text, speaker_id=0, noise_level=None, speech_rate=None, duration_noise_level=None, scale=None):

        if noise_level is None:
            noise_level = self.model.config["inference"].get("noise_level", 0.8)
        if speech_rate is None:
            speech_rate = self.model.config["inference"].get("speech_rate", 1.0)
        if duration_noise_level is None:
            duration_noise_level = self.model.config["inference"].get("duration_noise_level", 0.8)
        if scale is None:
            scale = self.model.config["inference"].get("scale", 1.0)

        text = text.strip()
        text = re.sub("—", "-", text)

        bert_embs = None
        phone_duration_extra = None

        bert = self.get_word_bert(text, nopunc=True)
        phoneme_ids, bert_embs = self.model.g2p.g2p_multistream(text, bert)
        bert_embs = np.expand_dims(np.transpose(np.array(bert_embs, dtype=np.float32)), 0)
        text = np.expand_dims(np.transpose(np.array(phoneme_ids, dtype=np.int64)), 0)
        text_lengths = np.array([text.shape[2]], dtype=np.int64)

        # Run main prediction
        scales = np.array([noise_level, 1.0 / speech_rate, duration_noise_level], dtype=np.float32)

        # Assign first voice
        if speaker_id is None:
            speaker_id = 0
        sid = np.array([speaker_id], dtype=np.int64)

        args = {
                "input": text,
                "input_lengths": text_lengths,
                "scales": scales,
                "sid": sid,
                "bert": bert_embs,
                "phone_duration_extra": phone_duration_extra,
        }

        start_time = time.perf_counter()
        audio = self.model.onnx.run(
            None,
            args
        )[0]
        audio = audio.squeeze()
        audio = audio * scale

        audio = self.audio_float_to_int16(audio)
        end_time = time.perf_counter()

        audio_duration_sec = audio.shape[-1] / 22050
        infer_sec = end_time - start_time
        real_time_factor = (
            infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        )

        logging.info("Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)" % (real_time_factor, infer_sec, audio_duration_sec))
        return audio
