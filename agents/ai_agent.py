from agents.base_agent import Base_Agent
import random
import numpy as np
import torch
from model_config import *


class AI_Agent(Base_Agent):
    def __init__(self, name, trainning_config=None):
        super().__init__()
        self.name = name
        self.trainning_config = trainning_config
        self.model = trainning_config["model"]
        model = model_config(**network_config)
    
    def choose_action(self, state, use_epsilon):
        print("我是AI智能体")
        if use_epsilon and np.random.rand() <= self.trainning_config["epsilon"]:
            return random.randrange(self.model.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def __str__(self):
        return f"AI_Agent({self.name})"
