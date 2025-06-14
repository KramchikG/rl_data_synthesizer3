from generators.hybrid_generator import HybridGenerator
from reward.reward_function import RewardFunction
from training.trainer import StudentTrainer
from rl.agent import PPOAgent
from generators.selection_strategy import SelectionStrategy
from prompt_sources.random_prompt_generator import RandomPromptGenerator  # <--- нове
from config import config
from logging_scr.logger import ExperimentLogger
import numpy as np
import logging

class ReinforcementLearningLoop:
    def __init__(self):
        self.agent = PPOAgent(
            state_dim=len(config.generation.generators),
            action_dim=len(config.generation.generators),
            lr=config.rl_agent.lr,
            gamma=config.rl_agent.gamma,
            clip_epsilon=config.rl_agent.epsilon,
            update_epochs=config.rl_agent.update_epochs
        )

        self.generator = HybridGenerator(agent=self.agent)
        self.reward_fn = RewardFunction()
        self.trainer = StudentTrainer()
        self.selector = SelectionStrategy(epsilon=config.rl_agent.epsilon)
        self.logger = ExperimentLogger()
        self.random_prompt_generator = RandomPromptGenerator()  # <--- нове

        self.reward_history = []
        self.epsilon_history = []
        self.generated_texts = set()  # Глобальний контроль унікальності

    def run_episode(self, prompt: str, reference: str, episode: int, min_words: int = 5):
        state = np.ones(len(config.generation.generators))
        max_attempts = 10

        for attempt in range(max_attempts):
            candidates, generation_trace, log_prob = self.generator.generate(
                prompt=prompt,
                state=state,
                return_trace=True,
                min_words=min_words,
                already_generated=self.generated_texts
            )

            if not candidates:
                logging.warning(f"[RL] No candidates generated in attempt {attempt + 1}/{max_attempts} for episode {episode}")
                continue

            candidate = candidates[0]
            chosen_generator_name = generation_trace.get(candidate, "unknown")

            self.generated_texts.add(candidate)

            reward = self.reward_fn.compute_reward(candidate, reference, generator_name=chosen_generator_name)

            self.logger.log("candidate", candidate, step=episode)
            self.logger.log("selected_generator", chosen_generator_name, step=episode)
            self.logger.log("selection_strategy", "ppo", step=episode)
            self.logger.log("reward", reward, step=episode)
            self.logger.log("prompt", prompt, step=episode)  # <--- нове

            # Враховує fallback або unknown генератор
            try:
                action_index = self.generator.selected_generators.index(chosen_generator_name)
            except ValueError:
                action_index = 0  # Безпечна заміна

            self.agent.update(
                states=[state],
                actions=[action_index],
                log_probs=[log_prob],
                rewards=[reward],
                next_states=[state],
                dones=[False]
            )

            self.reward_history.append(reward)
            avg_reward = np.mean(self.reward_history[-10:])
            self.logger.log("avg_reward", avg_reward, step=episode)
            self.logger.log("episode_done", True, step=episode)
            return  # Успішно завершено

        logging.warning(f"[RL] Skipped episode {episode} after {max_attempts} failed attempts.")
        self.logger.log("episode_skipped", True, step=episode)

    def run(self, num_episodes: int = 10):
        for episode in range(num_episodes):
            prompt = self.random_prompt_generator.generate_prompt()  # <--- нове
            reference = ""  # <--- тимчасово пусте
            self.run_episode(prompt, reference, episode)
