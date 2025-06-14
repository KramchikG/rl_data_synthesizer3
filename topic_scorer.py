#scoring/topic_scorer.py

from sentence_transformers import SentenceTransformer, util
import logging
from typing import List

class TopicScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, candidate: str, reference: str) -> float:

        try:
            embeddings = self.model.encode([reference, candidate], convert_to_tensor=True)
            return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
        except Exception as e:
            logging.warning(f"[TopicScorer] Semantic score failed: {e}")
            return 0.0

    def batch_score(self, candidates: List[str], reference: str) -> List[float]:

        try:
            embeddings = self.model.encode([reference] + candidates, convert_to_tensor=True)
            ref_embedding = embeddings[0].unsqueeze(0)
            candidate_embeddings = embeddings[1:]
            similarities = util.pytorch_cos_sim(ref_embedding, candidate_embeddings)[0]
            return similarities.cpu().tolist()
        except Exception as e:
            logging.warning(f"[TopicScorer] Batch score failed: {e}")
            return [0.0] * len(candidates)
