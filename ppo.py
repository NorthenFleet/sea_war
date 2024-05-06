import torch
import torch.nn as nn

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

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_units=128):
        super(PPO, self).__init__()
        self.body_network = BodyNetwork(input_dim, hidden_layers, hidden_units)
        self.policy_network = PolicyNetwork(self.body_network, output_dim)
        self.value_network = ValueNetwork(self.body_network)

    def forward(self, x):
        return self.policy_network(x), self.value_network(x)
    
     def save_model(self, name, episodes):
            file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode

