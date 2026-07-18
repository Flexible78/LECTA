import numpy as np
import json
import onnxruntime
from .char_tokenizer import CharTokenizer
from .text_postprocessor import fix_capital

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class AccentModel:
    def __init__(self) -> None:
        pass

    def load(self, path, device="cpu"):
        onnx_providers = onnxruntime.get_available_providers()
        providers = [p for p in onnx_providers if p in ["CUDAExecutionProvider", "CPUExecutionProvider"]]
        self.session = onnxruntime.InferenceSession(f"{path}/model.onnx", providers=providers)

        with open(f"{path}/config.json", "r") as f:
            self.id2label = json.load(f)["id2label"]
        self.tokenizer = CharTokenizer.from_pretrained(path)

    def render_stress(self, text, pred):
        text = list(text)
        i = 0
        for chunk in pred:
            if chunk['label'] != "NO" and chunk['label'] != "STRESS_SECONDARY" and chunk["score"] >= 0.55:
                text[i - 1] = "+" + text[i - 1]
            i += 1
        text = "".join(text)
        return text

    def put_accent(self, word):
        lower_word = word.lower()
        inputs = self.tokenizer(lower_word, return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        # Добавляем token_type_ids, если их нет
        if 'token_type_ids' not in inputs:
            seq_len = inputs['input_ids'].shape[1]
            inputs['token_type_ids'] = np.zeros((1, seq_len), dtype=np.int64)
        outputs = self.session.run(None, inputs)
        output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        logits = outputs[output_names["logits"]]
        probabilities = softmax(logits)
        scores = np.max(probabilities, axis=-1)[0]
        labels = np.argmax(logits, axis=-1)[0]
        pred_with_scores = [{'label': self.id2label[str(label)], 'score': float(score)} 
                            for label, score in zip(labels, scores)]

        stressed_word = self.render_stress(word, pred_with_scores)
        return stressed_word
