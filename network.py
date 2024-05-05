import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_units=[128, 128]):
        super(DQN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.ReLU())
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_units=[128, 128]):
        super(PPO, self).__init__()
        # 网络结构定义，类似于DQN的定义


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layers=2, hidden_units=[128, 128]):
        super(ActorCritic, self).__init__()
        # 网络结构定义，类似于DQN的定义


class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=2, hidden_units=[128, 128]):
        super(DDPG, self).__init__()
        # 网络结构定义，类似于DQN的定义


def configure_network(network_config):
    network_type = network_config.get("network_type", "dqn")
    input_dim = network_config.get("input_dim", 4)
    output_dim = network_config.get("output_dim", 2)
    hidden_layers = network_config.get("hidden_layers", 2)
    hidden_units = network_config.get("hidden_units", [128, 128])

    if network_type == "dqn":
        model = DQN(input_dim, output_dim, hidden_layers, hidden_units)
    elif network_type == "ppo":
        model = PPO(input_dim, output_dim, hidden_layers, hidden_units)
    elif network_type == "ac":
        model = ActorCritic(input_dim, output_dim, hidden_layers, hidden_units)
    elif network_type == "ddpg":
        model = DDPG(input_dim, output_dim, hidden_layers, hidden_units)
    else:
        raise NotImplementedError("Unsupported network type")

    optimizer = optim.Adam(model.parameters())

    return model, optimizer


# 使用示例
if __name__ == "__main__":
    network_config = {
        "network_type": "dqn",
        "input_dim": 10,
        "output_dim": 5,
        "hidden_layers": 3,
        "hidden_units": [256, 256, 256]
    }

    model, optimizer = configure_network(network_config)
    print(model)
