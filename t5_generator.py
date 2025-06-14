#generators/t5_generator.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List

class T5Generator:
    def __init__(self, model_name: str = "t5-small", max_length: int = 64):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length

    def generate(self, prompt: str, num_return_sequences: int = 3) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
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


