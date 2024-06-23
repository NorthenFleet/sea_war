class Agent():
    def __init__(self, env_config) -> None:
        network_config = {
            "max_entities": env_config["max_entities"],
            "max_tasks": env_config["max_tasks"],
            "entity_input_dim": env_config["entity_dim"],
            "task_input_dim": env_config["task_dim"],
            "entity_transformer_heads": 8,
            "task_transformer_heads": 8,
            "hidden_dim": 64,
            "num_layers": 1,
            "mlp_hidden_dim": 128,
            "entity_headds": env_config["max_entities"],
            "output_dim": env_config["max_tasks"]+1,  # max_tasks增加一个任务编号
            "transfer_dim": 128,
            "use_transformer": False,
            "use_head_mask": False
        }

    def choose_action(self, state):
        pass
