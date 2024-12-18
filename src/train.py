from env import Env
from init import Map, Weapon, Scenario
import numpy as np
import torch.optim as optim
from replay_bufer import ReplayBuffer


class Train():
    def __init__(self) -> None:
        # 环境
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/AirDefense.json'
        map_path = 'data/map.json'

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)

        self.game_config = {"scenario": scenario,
                            "map": map,
                            "weapon": weapon}
        self.current_step = None
        self.max_step = 1000
        self.game_env = Env(name, self.game_config)

        # 训练
        self.use_epsilon = True
        self.replay_buffer = ReplayBuffer(capacity=2000)

        self.training_config = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "model": "PPO",
            "input_dim": 100,
            "output_dim": 50
        }

        # 智能体设置，智能体数量与想定文件scenario一致
        player_config = {
            "red": ("agents.ai_agent", "AI_Agent", self.training_config),
            "blue": ("agents.rule_agent", "Rule_Agent")
        }

        self.players = {}
        for name, (module, cls, model) in player_config.items():
            player_class = getattr(__import__(module), cls)
            if model is not None:
                self.players[name] = player_class(name, model)
            else:
                self.players[name] = player_class(name)

    def run(self):
        obs = self.game_env.reset_game(self.game_config)
        done = False
        self.current_step = 0
        while not done:
            actions = {agent_name: agent.choose_action(
                obs, self.use_epsilon) for agent_name, agent in self.game_env.agents.items()}
            next_obs, rewards, done, info = self.game_env.update(
                actions)
            self.replay_buffer.push(obs, actions, rewards,
                                    next_obs, done)
            obs = next_obs

            self.current_step += 1
            if self.current_step > self.max_step:
                done = True
        print(self.current_step)


if __name__ == '__main__':
    train = Train()
    train.run()
