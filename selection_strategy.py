#generators/selection_strategy.py

import random
from typing import List, Tuple
from scoring.topic_scorer import TopicScorer
from reward.reward_function import RewardFunction


class SelectionStrategy:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.reward_function = RewardFunction()

    def select(self, candidates: List[Tuple[str, str]], reference: str, strategy: str = "epsilon_greedy", top_k: int = 3) -> List[str]:
        if not candidates:
            return []

        texts = [text for _, text in candidates]

        if strategy == "random":
            return [random.choice(texts)]

        elif strategy == "round_robin":
            return [texts[i % len(texts)] for i in range(min(top_k, len(texts)))]

        elif strategy == "score_top_k":
            scorer = TopicScorer()
            scores = scorer.batch_score(texts, reference)
            scored = list(zip(texts, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [text for text, _ in scored[:top_k]]

        elif strategy == "epsilon_greedy":
            # Обчислюємо винагороду для кожного кандидата
            rewards = [self.reward_function.compute_reward(text, reference) for text in texts]

            if random.random() < self.epsilon:
                return [random.choice(texts)]
            else:
                best_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
                return [texts[best_idx]]

        elif strategy == "weighted":
            weights = [random.random() for _ in texts]
            selected = random.choices(texts, weights=weights, k=1)
            return selected

        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
