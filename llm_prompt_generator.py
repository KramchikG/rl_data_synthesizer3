from transformers import pipeline

class LLMInstructionGenerator:
    def __init__(self, model_name="gpt2", max_length=50, num_return_sequences=1):
        self.generator = pipeline("text-generation", model=model_name)
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences

        self.seeds = [
            "Give a detailed instruction to",
            "Describe a procedure to",
            "Explain how to",
            "Write a prompt asking to",
            "How would you instruct someone to"
        ]

    def generate_synthetic_prompts(self, num_prompts: int):
        prompts = []
        for _ in range(num_prompts):
            seed = random.choice(self.seeds)
            result = self.generator(
                seed,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                do_sample=True,
                temperature=0.9,
                top_k=50
            )
            prompts.append(result[0]['generated_text'].strip())
        return prompts


def generate_synthetic_prompts(num_prompts=5):
    generator = LLMInstructionGenerator()
    return generator.generate_synthetic_prompts(num_prompts)
