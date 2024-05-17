from agents.base_agent import Base_Agent
import random
import numpy as np
import torch
from model_config import *


class AI_Agent(Base_Agent):
    def __init__(self, AI_config):
        super().__init__()
        self.name = AI_config["name"]
        self.AI_config = AI_config
        self.model = AI_config["model"]
        model = model_config(**self.model)

        # 动态导入智能体模块
        # self.players = {name: getattr(__import__(module), cls)(
        #     name) for name, (module, cls) in player_config.items()}

        self.agents = {}
        for name, (module, cls, training_config) in ai_config.items():
            agent_class = getattr(__import__(module), cls)
            if training_config["model"] is not None:
                self.players[name] = agent_class(name, training_config)
            else:
                self.players[name] = agent_class(name)

        
        self.model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(
        self.model.parameters(), lr=self.learning_rate)
    
    def choose_action(self, state, use_epsilon):
        print("我是AI智能体")
        if use_epsilon and np.random.rand() <= self.trainning_config["epsilon"]:
            return random.randrange(self.model.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def __str__(self):
        return f"AI_Agent({self.name})"
