#generators/formalization_generator.py

from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class FormalizationGenerator:
    def __init__(self, model_name="google/flan-t5-small", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:
        input_text = f"Rewrite in formal style: {prompt}"
        inputs = self.tokenizer([input_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            num_return_sequences=num_return_sequences
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
