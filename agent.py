
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, gamma: float = 0.99, clip_epsilon: float = 0.2, update_epochs: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        self.actor = self.build_network(output_dim=action_dim)
        self.critic = self.build_network(output_dim=1)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def build_network(self, output_dim: int):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def select_action(self, state: np.ndarray) -> (int, torch.Tensor):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def compute_advantage(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, log_probs, rewards, next_states, dones):
        # Перетворення до тензорів
        states_tensor = torch.FloatTensor(np.array(states)).detach().clone()
        next_states_tensor = torch.FloatTensor(np.array(next_states)).detach().clone()

        # Обчислення values та next_values
        values = self.critic(states_tensor).squeeze().detach().clone()
        next_values = self.critic(next_states_tensor).squeeze().detach().clone()

        # Переконаємось, що `values` та `next_values` є 1D тензорами
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if next_values.dim() == 0:
            next_values = next_values.unsqueeze(0)

        # Від'єднуємо граф обчислень для `values` та `next_values`
        values_detached = values.clone()
        next_values_detached = next_values.clone()

        # Обчислюємо переваги (advantages)
        advantages = self.compute_advantage(
            rewards,
            values_detached.numpy(),
            next_values_detached.numpy(),
            dones
        )
        advantages = torch.FloatTensor(advantages).detach().clone()

        # Конвертуємо решту даних до тензорів
        log_probs = torch.FloatTensor(log_probs).detach().clone()
        rewards = torch.FloatTensor(rewards).detach().clone()
        actions = torch.LongTensor(actions).detach().clone()

        # Основний цикл оновлення
        for epoch in range(self.update_epochs):
            logits = self.actor(states_tensor)  # без .detach()
            probs = torch.softmax(logits, dim=-1)

            new_log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze())

            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            values_pred = self.critic(states_tensor).squeeze()
            value_loss = nn.MSELoss()(values_pred, rewards)

            loss = actor_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Обчислює ймовірності дій (тобто генераторів) на основі поточного стану.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape: [1, state_dim]
        with torch.no_grad():
            logits = self.actor(state_tensor)  # shape: [1, action_dim]
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: [action_dim]
        return probs.cpu().numpy()
