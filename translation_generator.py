#generators/translation_generator.py

from typing import List
from transformers import MarianMTModel, MarianTokenizer
import torch

class TranslationGenerator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-fr", device=None):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        translated = self.model.generate(
            **inputs,
            num_return_sequences=num_return_sequences,
            max_new_tokens=50,  # critical
            num_beams=4
        )
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
