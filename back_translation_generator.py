#generators/back_translation_generator.py

from transformers import MarianMTModel, MarianTokenizer
from typing import List

class BackTranslationGenerator:
    def __init__(self,
                 src_lang: str = "en",
                 pivot_lang: str = "de",  # Pivot: German
                 model_src_to_pivot: str = "Helsinki-NLP/opus-mt-en-de",
                 model_pivot_to_src: str = "Helsinki-NLP/opus-mt-de-en",
                 max_length: int = 64):
        self.tokenizer_fwd = MarianTokenizer.from_pretrained(model_src_to_pivot)
        self.model_fwd = MarianMTModel.from_pretrained(model_src_to_pivot)
        self.tokenizer_back = MarianTokenizer.from_pretrained(model_pivot_to_src)
        self.model_back = MarianMTModel.from_pretrained(model_pivot_to_src)
        self.max_length = max_length

    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:
        # DE
        encoded = self.tokenizer_fwd(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        translated = self.model_fwd.generate(**encoded, num_return_sequences=num_return_sequences)
        pivot_texts = [self.tokenizer_fwd.decode(t, skip_special_tokens=True) for t in translated]

        # EN
        results = []
        for pivot in pivot_texts:
            encoded_back = self.tokenizer_back(pivot, return_tensors="pt", truncation=True, max_length=self.max_length)
            back_translated = self.model_back.generate(**encoded_back)
            back_text = self.tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
            results.append(back_text)

        return results
