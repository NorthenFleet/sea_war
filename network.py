import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, layers):
        super(DQN, self).__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x)


class PPO(nn.Module):
    def __init__(self, layers):
        super(PPO, self).__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x)

class ActorCritic(nn.Module):
    def __init__(self, layers):
        super(ActorCritic, self).__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x)

class DDPG(nn.Module):
    def __init__(self, layers):
        super(DDPG, self).__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x) 

# def configure_network(config):
#     input_dim = config.get("input_dim", 4)
#     output_dim = config.get("output_dim", 2)
#     hidden_layers = config.get("hidden_layers", 2)
#     hidden_units = config.get("hidden_units", [128, 128])

def configure_network(model_type, input_dim, output_dim, hidden_layers=2, hidden_units=[128, 128]):
    if model_type == "DQN":
        model = DQN(layers=build_layers(input_dim, output_dim, hidden_layers, hidden_units))
    elif model_type == "PPO":
        model = PPO(layers=build_layers(input_dim, output_dim, hidden_layers, hidden_units))
    elif model_type == "ActorCritic":
        model = ActorCritic(layers=build_layers(input_dim, output_dim, hidden_layers, hidden_units))
    elif model_type == "DDPG":
        model = DDPG(layers=build_layers(input_dim, output_dim, hidden_layers, hidden_units))
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
        "model_type":"DQN",
        "input_dim": 10,
        "output_dim": 5,
        "hidden_layers": 3,
        "hidden_units": [256, 256, 256]
    }

    model = configure_network(**network_config)
    print(model)