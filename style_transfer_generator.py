# generators/style_transfer.py

from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class StyleTransferGenerator:
    def __init__(self, model_name="prithivida/parrot_paraphraser_on_T5", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:
        input_text = f"paraphrase: {prompt} </s>"
        inputs = self.tokenizer([input_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            num_return_sequences=num_return_sequences,
            temperature=1.5
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
