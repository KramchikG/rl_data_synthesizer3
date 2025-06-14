#generators/base_generator.py

from typing import List
from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:

        pass
