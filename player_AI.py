from player_base import Player_Base
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model_select import *


class AIPlayer(Player_Base):
    def __init__(self, AI_config):
        super(AIPlayer, self).__init__()
        self.state_size = AI_config["state_size"]
        self.action_size = AI_config["action_size"]
        
        config = {
            "model_type": AI_config["model"],
            "input_dim": self.state_size,
            "output_dim": self.action_size
        }

        self.modle = model_select(**config)
        self.use_epsilon = AI_config["use_epsilon"]
        self.epsilon = AI_config["epsilon"]
        self.epsilon_min = AI_config["epsilon_min"]
        self.epsilon_decay = AI_config["epsilon_decay"]
        self.learning_rate = AI_config["learning_rate"]
        self.agents = {}
        self.memory = []

    def choose_action(self, state):
        print("我是AI智能体")

        if self.use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return np.argmax(act_values.detach().numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, samples):
        for state, action, reward, next_state, done in samples:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy()))
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
