from agents.base_agent import Base_Agent
import random
import numpy as np
import torch


class AI_Agent(Base_Agent):
    def __init__(self, name, model=None):
        super().__init__(name)
        self.model = model

    
    def choose_action(self, state, use_epsilon=True):
            print("我是AI智能体")
            return random.choice(self.model.output_dim)
            # if use_epsilon and np.random.rand() <= self.epsilon:
            #     return random.randrange(self.action_size)
            # state = torch.FloatTensor(state).unsqueeze(0)
            # act_values = self.model(state)
            # return np.argmax(act_values.detach().numpy())



    def __str__(self):
        return f"AI_Agent({self.name})"
