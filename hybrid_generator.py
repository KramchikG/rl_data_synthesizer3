from typing import List, Tuple, Dict, Set
from config import config
import random
import numpy as np
from generators import (
    t5_generator,
    gpt2_generator,
    back_translation_generator,
    style_transfer_generator,
    translation_generator,
    formalization_generator,
)
from rl.agent import PPOAgent
from langdetect import detect, DetectorFactory
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DetectorFactory.seed = 0

def is_english(text: str) -> bool:
    try:
        return detect(text) == 'en'
    except Exception as e:
        logging.warning(f"[HybridGenerator] Language detection failed: {e}")
        return False

def deduplicate_similar_texts(texts: List[str], threshold: float = 0.95) -> List[str]:
    if len(texts) <= 1:
        return texts

    vectorizer = TfidfVectorizer().fit_transform(texts)
    similarity_matrix = cosine_similarity(vectorizer)

    keep_indices = []
    seen = set()

    for i in range(len(texts)):
        if i in seen:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i, j] >= threshold:
                seen.add(j)

    return [texts[i] for i in keep_indices]

class HybridGenerator:
    def __init__(self, agent: PPOAgent = None):
        self.generators_map = {
            "t5_generator": t5_generator.T5Generator(),
            "gpt2_generator": gpt2_generator.GPT2Generator(),
            "back_translation_generator": back_translation_generator.BackTranslationGenerator(),
            "style_transfer_generator": style_transfer_generator.StyleTransferGenerator(),
            "translation_generator": translation_generator.TranslationGenerator(),
            "formalization_generator": formalization_generator.FormalizationGenerator(),
        }
        self.selected_generators = config.generation.generators
        self.return_all = config.generation.return_all
        self.agent = agent

    def generate(
            self,
            prompt: str,
            state: np.ndarray,
            return_trace: bool = False,
            min_words: int = 5,
            already_generated: Set[str] = None,
            max_generators_sampled: int = 3  # Кількість генераторів на один епізод
    ) -> Tuple[List[str], Dict[str, str], float]:
        candidates = []
        trace_map = {}

        if already_generated is None:
            already_generated = set()

        # Обчислити ймовірності для вибору генераторів з урахуванням PPO або випадково
        if self.agent:
            action_probs = self.agent.get_action_probabilities(state)
        else:
            action_probs = np.ones(len(self.selected_generators)) / len(self.selected_generators)

        # Вибір підмножини генераторів
        try:
            sampled_generator_indices = np.random.choice(
                len(self.selected_generators),
                size=min(max_generators_sampled, len(self.selected_generators)),
                replace=False,
                p=action_probs
            )
        except ValueError:
            logging.warning("[HybridGenerator] Invalid probability distribution. Falling back to uniform.")
            sampled_generator_indices = np.random.choice(
                len(self.selected_generators),
                size=min(max_generators_sampled, len(self.selected_generators)),
                replace=False
            )

        sampled_generator_names = [self.selected_generators[i] for i in sampled_generator_indices]

        for name in sampled_generator_names:
            generator = self.generators_map[name]
            try:
                samples = generator.generate(prompt, num_return_sequences=config.generation.max_outputs_per_prompt)
                for sample in samples:
                    if sample is None:
                        continue
                    if len(sample.strip().split()) < min_words:
                        continue
                    if not is_english(sample):
                        continue
                    if sample in already_generated:
                        continue

                    candidates.append(sample)
                    trace_map[sample] = name
            except Exception as e:
                logging.warning(f"[HybridGenerator] Generator '{name}' failed: {e}")
                continue

        filtered_candidates = deduplicate_similar_texts(candidates, threshold=0.95)

        if not filtered_candidates:
            fallback = f"Fallback response for: {prompt}"
            trace_map[fallback] = "fallback_generator"
            return ([fallback], trace_map, 0.0)

        action_idx, log_prob = self.agent.select_action(state)
        selected_generator = self.selected_generators[action_idx]

        final_candidates = [s for s in filtered_candidates if trace_map.get(s) == selected_generator]

        if not final_candidates:
            fallback = random.choice(filtered_candidates)
            return ([fallback], trace_map, log_prob)

        selected_sample = random.choice(final_candidates)
        return ([selected_sample], trace_map, log_prob) if return_trace else ([selected_sample], {}, log_prob)
