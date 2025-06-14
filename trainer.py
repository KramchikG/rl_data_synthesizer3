from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
import torch
from datasets import Dataset
from typing import List, Tuple

class StudentTrainer:
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_type = self._detect_model_type(model_name)
        self.model = self._load_model(model_name)

    def _detect_model_type(self, model_name: str) -> str:
        """Розпізнає тип моделі на основі її імені."""
        name = model_name.lower()
        if "t5" in name or "mbart" in name:
            return "seq2seq"
        elif "gpt" in name or "opt" in name or "bloom" in name:
            return "causal"
        elif "bert" in name or "electra" in name:
            return "masked"
        else:
            return "seq2seq"  # за замовчуванням

    def _load_model(self, model_name: str):
        """Завантажує модель відповідно до її типу."""
        if self.model_type == "seq2seq":
            return AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif self.model_type == "causal":
            return AutoModelForCausalLM.from_pretrained(model_name)
        elif self.model_type == "masked":
            return AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model type for {model_name}")

    def _prepare_dataset(self, data: List[Tuple[str, str]]) -> Dataset:
        examples = [{"input": inp, "target": tgt} for inp, tgt in data]

        def tokenize_seq2seq(example):
            model_inputs = self.tokenizer(
                example["input"], max_length=128, padding="max_length", truncation=True
            )
            labels = self.tokenizer(
                example["target"], max_length=128, padding="max_length", truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def tokenize_causal(example):
            combined = example["input"] + self.tokenizer.eos_token + example["target"]
            tokenized = self.tokenizer(combined, max_length=128, padding="max_length", truncation=True)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        def tokenize_masked(example):
            tokenized = self.tokenizer(
                example["input"], max_length=128, padding="max_length", truncation=True
            )
            tokenized["labels"] = self.tokenizer(
                example["target"], max_length=128, padding="max_length", truncation=True
            )["input_ids"]
            return tokenized

        dataset = Dataset.from_list(examples)

        if self.model_type == "seq2seq":
            return dataset.map(tokenize_seq2seq, remove_columns=["input", "target"])
        elif self.model_type == "causal":
            return dataset.map(tokenize_causal, remove_columns=["input", "target"])
        elif self.model_type == "masked":
            return dataset.map(tokenize_masked, remove_columns=["input", "target"])
        else:
            raise ValueError("Unknown model type")

    def train_on(self, example: str, target: str = "placeholder"):
        self.train_on_dataset([(example, target)], num_train_epochs=1)

    def train_on_dataset(
        self,
        data: List[Tuple[str, str]],
        output_dir: str = "saved_models/student",
        num_train_epochs: int = 3,
        batch_size: int = 8
    ):
        dataset = self._prepare_dataset(data)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_strategy="no",
            remove_unused_columns=False
        )

        # Вибір відповідного data collator
        if self.model_type == "seq2seq":
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        elif self.model_type == "causal":
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        elif self.model_type == "masked":
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()

    def save_model(self, path: str = "saved_models/student"):
        self.model.save_pretrained(path)

    def load_model(self, path: str = "saved_models/student"):
        self.model = self._load_model(path)
