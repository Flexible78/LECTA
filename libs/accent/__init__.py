import re
import requests
import torch
import gc
from .ruaccent import RUAccent
from pathlib import Path
from libs.utils import download_model, models_path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Run with {device}')

class ACCModel:
    def __init__(self):
        self.accentizer = None
        self.ver = None

    def load(self, ver):
        if self.accentizer is not None:
            del self.accentizer
            self.accentizer = None
            gc.collect()

        self.ver = ver

        if ver == 1:
            self.accentizer = RUAccent()
            self.accentizer.load(
                omograph_model_size='turbo3.1',
                use_dictionary=True,
                device=device.upper(),
                workdir=str( models_path / "RuAccent"))
        else:
            model_path = models_path / "silero_stress" / "accentor.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if not model_path.is_file():
                print("Loading Silero Stress model...")
                model_url = "https://github.com/snakers4/silero-stress/raw/refs/heads/master/src/silero_stress/data/accentor.pt"
                m, status = download_model(model_url, model_path)
                if m is None:
                    return m, status
            try:
                package = torch.package.PackageImporter(str(model_path))
                self.accentizer = package.load_pickle("accentor_models", "accentor")
                weights = self.accentizer.homosolver.model.bert.embeddings.word_embeddings.weight.data
                scale = self.accentizer.homosolver.model.bert.scale
                zero_point = self.accentizer.homosolver.model.bert.zero_point
                restored = scale * (weights - zero_point)
                self.accentizer.homosolver.model.bert.embeddings.word_embeddings.weight.data = restored
            except Exception as e:
                return None, f"Model loading error: {e}"

        return ver, "Model loaded successfully!"

    def process_accent(self, string, regexp):
        if self.ver == 1:
            return self.accentizer.process_all(string, regexp)
        if not regexp:
            return self.accentizer(string)
        
        pattern = re.compile(regexp)
        matches = list(pattern.finditer(string))
        
        if not matches:
            return self.accentizer(string)

        result_parts = []
        prev_end = 0
        
        for match in matches:
            start, end = match.start(), match.end()
            result_parts.append(self.accentizer(string[prev_end:start]))
            result_parts.append(string[start:end])
            
            prev_end = end
        
        result_parts.append(self.accentizer(string[prev_end:]))
        
        return "".join(result_parts)

accentizer = ACCModel()
