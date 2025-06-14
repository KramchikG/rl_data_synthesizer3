class Config:
    def __init__(self):
        self.generation = self.Generation()
        self.scoring = self.Scoring()
        self.rl_agent = self.RLAgent()
        self.reward = self.Reward()
        self.training = self.Training()
        self.logging = self.Logging()
        self.filtering = self.Filtering()

    class Generation:

        generators = [
            "t5_generator",
            "gpt2_generator",
            "back_translation_generator",
            "style_transfer_generator",
            "translation_generator",
            "formalization_generator"
        ]
        hybrid_strategy = "epsilon_greedy"
        return_all = False
        max_outputs_per_prompt = 3
        prompt_source = "random"  # або "gpt2"
        num_prompts = 5

    class Scoring:
        topic_reference = "movies, film, review"
        similarity_weight = 0.7
        diversity_weight = 0.3

    class RLAgent:
        lr = 1e-4
        gamma = 0.99
        epsilon = 0.2
        update_epochs = 4

    class Reward:
        use_bleu = True
        use_novelty = True
        use_length_penalty = False
        aggregation = "weighted_sum"

    class Training:
        model_name = "t5-small"
        learning_rate = 1e-4
        num_episodes = 100000

    class Logging:
        log_dir = "logs/"
        save_json = True
        save_csv = True
        use_mlflow = False

    class Filtering:
        enable_domain_filter = True
        allowed_keywords = ["movie", "film", "cinema", "review"]
        enabled = False
        domain = "translation"

config = Config()
