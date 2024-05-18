import torch
import torch.nn as nn
import torch.optim as optim


class BodyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers=2, hidden_units=128):
        super(BodyNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PolicyNetwork(nn.Module):
    def __init__(self, body_network, output_dim):
        super(PolicyNetwork, self).__init__()
        self.body_network = body_network
        self.head = nn.Linear(body_network.layers[-2].out_features, output_dim)

    def forward(self, x):
        features = self.body_network(x)
        return torch.softmax(self.head(features), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, body_network):
        super(ValueNetwork, self).__init__()
        self.body_network = body_network
        self.head = nn.Linear(body_network.layers[-2].out_features, 1)

    def forward(self, x):
        features = self.body_network(x)
        return self.head(features)


class DDPG():
    def __init__(self, input_dim, output_dim, lr=0.001):
        super(DDPG, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        config = {
            "input_dim": input_dim,
            "hidden_layers": 2,
            "hidden_units": 128
        }
        self.body_network = BodyNetwork(**config)
        self.policy_network = PolicyNetwork(self.body_network, output_dim)
        self.value_network = ValueNetwork(self.body_network)

        # 初始化优化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.policy_network(x), self.value_network(x)
