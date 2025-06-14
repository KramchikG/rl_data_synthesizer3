# scoring/semantic_scorer.py

from sentence_transformers import SentenceTransformer, util
from typing import List

class SemanticScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, candidate: str, reference: str) -> float:
        try:
            emb_ref = self.model.encode(reference, convert_to_tensor=True)
            emb_cand = self.model.encode(candidate, convert_to_tensor=True)
            return float(util.cos_sim(emb_ref, emb_cand)[0][0])
        except Exception:
            return 0.0

    def batch_score(self, candidates: List[str], reference: str) -> List[float]:
        try:
            emb_ref = self.model.encode(reference, convert_to_tensor=True)
            emb_cands = self.model.encode(candidates, convert_to_tensor=True)
            sims = util.cos_sim(emb_ref, emb_cands)[0]
            return sims.tolist()
        except Exception:
            return [0.0] * len(candidates)
