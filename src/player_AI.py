from player import Player
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model_select import *


class AIPlayer(Player):
    def __init__(self):
        super(AIPlayer, self).__init__()

        agent_config = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "model": "PPO",
            "state_size": 100,
            "action_size": 50,
            "use_epsilon": True,
        }

        self.state_size = agent_config["state_size"]
        self.action_size = agent_config["action_size"]

        config = {
            "model_type": agent_config["model"],
            "input_dim": self.state_size,
            "output_dim": self.action_size
        }

        self.modle = model_select(**config)
        self.use_epsilon = agent_config["use_epsilon"]
        self.epsilon = agent_config["epsilon"]
        self.epsilon_min = agent_config["epsilon_min"]
        self.epsilon_decay = agent_config["epsilon_decay"]
        self.learning_rate = agent_config["learning_rate"]
        self.agents = {}
        self.memory = []

    def choose_action(self, state):
        print("我是AI智能体")

        if self.use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return np.argmax(act_values.detach().numpy())

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def train(self, samples):
        for state, action, reward, next_state, done in samples:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy()))
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.model.zero_grad()
            loss = nn.MSELoss()(target_f, torch.FloatTensor([target]))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode


if __name__ == '__main__':

    AI = AIPlayer()
    AI.save_model("ppo", "000")
    print("保存网络")
