#generators/gpt2_generator.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List
import torch

class GPT2Generator:
    def __init__(self, model_name: str = "gpt2", max_length: int = 64):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.max_length = max_length

    def generate(self, prompt: str, num_return_sequences: int = 3) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

