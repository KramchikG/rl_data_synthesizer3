#ligging_scr/logger.py

import os
import json
import csv
from typing import Any

class ExperimentLogger:
    def __init__(self, base_dir: str = "logs/"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.json_log_path = os.path.join(self.base_dir, "log.jsonl")
        self.csv_log_path = os.path.join(self.base_dir, "log.csv")

        # create empty files if they don't exist
        if not os.path.exists(self.json_log_path):
            open(self.json_log_path, 'w', encoding='utf-8').close()
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "key", "value"])

    def log(self, key: str, value: Any, step: int):
        json_entry = {"step": step, "key": key, "value": value}
        with open(self.json_log_path, "a", encoding='utf-8') as f_json:
            f_json.write(json.dumps(json_entry, ensure_ascii=False) + "\n")

        with open(self.csv_log_path, "a", newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([step, key, value])
