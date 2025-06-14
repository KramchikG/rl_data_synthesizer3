import logging
from reward.reward_logger import RewardLogger
from scoring.topic_scorer import TopicScorer
from sentence_transformers import SentenceTransformer, util
import numpy as np

class RewardFunction:
    def __init__(self,
                 use_topic: bool = True,
                 use_novelty: bool = True,
                 use_length_penalty: bool = True,
                 target_length: int = 50,
                 similarity_weight: float = 1.0,
                 novelty_weight: float = 1.0,
                 length_weight: float = 0.1):
        self.topic_scorer = TopicScorer()
        self.semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.logger = RewardLogger()

        self.use_topic = use_topic
        self.use_novelty = use_novelty
        self.use_length_penalty = use_length_penalty
        self.target_length = target_length

        self.similarity_weight = similarity_weight
        self.novelty_weight = novelty_weight
        self.length_weight = length_weight

        self.memory = set()
        self.memory_embeddings = []

        # Для згладженого моніторингу винагород
        self.exp_average_reward = 0.0
        self.exp_beta = 0.9

    def _semantic_similarity(self, candidate: str, reference: str) -> float:
        try:
            emb = self.semantic_model.encode([candidate, reference], convert_to_tensor=True)
            return float(util.pytorch_cos_sim(emb[0], emb[1]))
        except Exception as e:
            logging.warning(f"[RewardFunction] Semantic similarity failed: {e}")
            return 0.0

    def _embedding_novelty(self, candidate: str) -> float:
        try:
            emb = self.semantic_model.encode(candidate, convert_to_tensor=True)
            if not self.memory_embeddings:
                self.memory_embeddings.append(emb)
                return 1.0
            similarities = util.pytorch_cos_sim(emb, self.memory_embeddings)[0]
            max_sim = float(similarities.max())
            self.memory_embeddings.append(emb)
            return 1.0 - max_sim
        except Exception as e:
            logging.warning(f"[RewardFunction] Embedding novelty failed: {e}")
            return 0.0

    def _jaccard_novelty(self, candidate: str) -> float:
        candidate_set = set(candidate.split())
        if not self.memory:
            return 1.0
        similarities = [
            len(candidate_set & set(prev.split())) / len(candidate_set | set(prev.split()))
            for prev in self.memory
        ]
        return 1.0 - max(similarities)

    def _length_score(self, candidate: str) -> float:
        length_diff = abs(len(candidate.split()) - self.target_length)
        return np.exp(-length_diff / self.target_length)

    def _add_component(self, name: str, score: float, weight: float, reward_accumulator: dict):
        contribution = weight * score
        reward_accumulator['reward'] += contribution
        reward_accumulator['logs'][f'{name}_score'] = score
        reward_accumulator['logs'][f'{name}_contribution'] = contribution

    def compute_reward(self, candidate: str, reference: str, generator_name: str = "unknown") -> float:
        reward_acc = {'reward': 0.0, 'logs': {'generator': generator_name}}

        # --- Topic + Semantic Similarity ---
        if self.use_topic:
            topic_score = self.topic_scorer.score(candidate, reference)
            semantic_score = self._semantic_similarity(candidate, reference)
            alpha = 0.7 if topic_score < 0.3 and semantic_score > 0.7 else 0.5
            combined_similarity = alpha * topic_score + (1 - alpha) * semantic_score

            self._add_component('similarity', combined_similarity, self.similarity_weight, reward_acc)
            reward_acc['logs'].update({
                'topic_score': topic_score,
                'semantic_score': semantic_score,
                'alpha': alpha
            })

        # --- Новизна ---
        if self.use_novelty:
            jaccard_score = self._jaccard_novelty(candidate)
            embedding_score = self._embedding_novelty(candidate)
            combined_novelty = 0.5 * jaccard_score + 0.5 * embedding_score
            self.memory.add(candidate)
            self._add_component('novelty', combined_novelty, self.novelty_weight, reward_acc)
            reward_acc['logs'].update({
                'jaccard_novelty': jaccard_score,
                'embedding_novelty': embedding_score,
                'combined_novelty': combined_novelty,
                'novelty_weight_before': self.novelty_weight
            })

            # --- Адаптивне оновлення ваги новизни ---
            novelty_threshold = 0.3
            max_weight = 2.0
            min_weight = 0.1
            adjust_step = 0.05

            if combined_novelty < novelty_threshold:
                self.novelty_weight = min(max_weight, self.novelty_weight + adjust_step)
            else:
                self.novelty_weight = max(min_weight, self.novelty_weight - adjust_step * 0.5)

            reward_acc['logs']['novelty_weight_after'] = self.novelty_weight

        # --- Пеналізація за довжину ---
        if self.use_length_penalty:
            length_score = self._length_score(candidate)
            self._add_component('length', length_score, self.length_weight, reward_acc)

        # --- Згладжене середнє винагород ---
        self.exp_average_reward = (
            self.exp_beta * self.exp_average_reward + (1 - self.exp_beta) * reward_acc['reward']
        )

        # --- Логування додаткових метрик ---
        reward_acc['logs']['total_reward'] = reward_acc['reward']
        reward_acc['logs']['exp_average_reward'] = self.exp_average_reward
        reward_acc['logs']['novelty_memory_size'] = len(self.memory)
        reward_acc['logs']['embedding_memory_size'] = len(self.memory_embeddings)
        reward_acc['logs']['candidate'] = candidate[:80] + '...' if len(candidate) > 80 else candidate

        # --- Вивід логів ---
        logging.info(f"[RewardFunction] Reward breakdown: {reward_acc['logs']}")
        self.logger.log(reward_acc['logs'], generator_name=generator_name)

        return reward_acc['reward']
