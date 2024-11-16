import torch.nn as nn
from models.ppo import PPO
from models.dqn import DQN
from models.actor_critic import ActorCritic
from models.ddpg import DDPG

# def configure_network(config):
#     input_dim = config.get("input_dim", 4)
#     output_dim = config.get("output_dim", 2)
#     hidden_layers = config.get("hidden_layers", 2)
#     hidden_units = config.get("hidden_units", [128, 128])


def model_select(model_type, input_dim, output_dim):
    if model_type == "DQN":
        model = DQN(input_dim, output_dim)
    elif model_type == "PPO":
        model = PPO(input_dim, output_dim)
    elif model_type == "DDPG":
        model = DDPG(input_dim, output_dim)
    elif model_type == "AC":
        model = ActorCritic(input_dim, output_dim)
    else:
        raise NotImplementedError("Unsupported model type")

    return model


def build_layers(input_dim, output_dim, hidden_layers, hidden_units):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_units[0]))
    layers.append(nn.ReLU())
    for i in range(1, hidden_layers):
        layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_units[-1], output_dim))
    return nn.Sequential(*layers)


# 使用示例
if __name__ == "__main__":
    network_config = {
        "model_type": "DQN",
        "input_dim": 10,
        "output_dim": 5,
        "hidden_layers": 3,
        "hidden_units": [256, 256, 256]
    }

    model = model_config(**network_config)
    print(model)
