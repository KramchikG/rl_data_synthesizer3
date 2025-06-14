from config import config
from rl.loop import ReinforcementLearningLoop

import transformers
import torch
import logging

# Додатково: встановлення флагів для дебагу та усунення шуму логів трансформерів
torch.autograd.set_detect_anomaly(True)
transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO)

def main():
    # Повністю ізольований ReinforcementLearningLoop управляє генерацією промптів
    rl_loop = ReinforcementLearningLoop()
    rl_loop.run(num_episodes=config.training.num_episodes)

if __name__ == "__main__":
    main()
