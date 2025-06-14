# reward/reward_logger.py
import json
import os
from typing import Optional

class RewardLogger:
    def __init__(self, base_dir: str = "logs/reward_logs/"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.general_log_path = os.path.join(self.base_dir, "all_rewards.jsonl")

    def log(self, log_data: dict, generator_name: Optional[str] = None):
        """
        Логує винагороду в загальний файл, а також в окремий файл, якщо вказано генератор.
        """
        # Лог в загальний файл
        with open(self.general_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        # Лог в генератор-специфічний файл (опційно)
        if generator_name:
            path = os.path.join(self.base_dir, f"{generator_name}_rewards.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
