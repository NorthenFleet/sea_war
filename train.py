from env import GameEnv
from init import Map, Weapon, Scenario
import numpy as np
import torch.optim as optim
from replay_bufer import ReplayBuffer
from model_config import *


class Train():
    def __init__(self) -> None:
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/scenario.json'
        map_path = 'data/map.json'

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)

        # 环境
        self.input_dim = 10
        self.output_dim = 5
        self.game_config = {"scenario": scenario,
                       "map": map,
                       "weapon": weapon}
        self.current_step = None
        self.max_step = 1000

        # 训练
        network_config = {
            "model_type": "PPO",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }
        self.model = model_config(**network_config)
        self.use_epsilon = True
        self.replay_buffer = ReplayBuffer(capacity=2000)

        self.training_config = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001
        }

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.training_config["learning_rate"])

        # 智能体
        agent_modules = {
            "agent1": ("agents.ai_agent", "AI_Agent", self.training_config, self.model),
            "agent2": ("agents.rule_agent", "Rule_Agent", None, None)
        }
        self.game_env = GameEnv(name, agent_modules)

    def run(self):
        observation = self.game_env.reset_game(self.game_config)
        done = False
        self.current_step = 0
        while not done:
            actions = {agent_name: agent.choose_action(
                observation, self.use_epsilon) for agent_name, agent in self.game_env.agents.items()}
            observations, rewards, done, info = self.game_env.update(
                actions)
            next_observations = np.reshape(
                next_observations, [1, self.input_dim])
            self.replay_buffer.push(observations, actions, rewards,
                                    next_observations, done)

            self.current_step += 1
            if self.current_step > self.max_step:
                done = True
        print(self.current_step)


if __name__ == '__main__':
    train = Train()
    train.run()
