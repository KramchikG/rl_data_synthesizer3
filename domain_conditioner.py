#filtering/domain_conditioner
from typing import List

class DomainConditioner:
    def __init__(self, allowed_keywords: List[str]):
        self.allowed_keywords = [kw.lower() for kw in allowed_keywords]

    def filter_examples(self, examples: List[str]) -> List[str]:
        filtered = []
        for ex in examples:
            if any(kw in ex.lower() for kw in self.allowed_keywords):
                filtered.append(ex)
        return filtered

    def is_relevant(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.allowed_keywords)
