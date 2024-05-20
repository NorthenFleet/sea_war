import ray
from env import Env
from init import Map, Weapon, Scenario
import numpy as np
import torch.optim as optim
from replay_bufer import ReplayBuffer


# @ray.remote
class DistributedGameEnv:
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.game_env = Env(self.config["game_config"])
        self.replay_buffer = ReplayBuffer(
            self.config["trainning_config"]["buffer_capacity"])
        self.players = {}
        for name, (path, module, config) in self.config["player_config"].items():
            player_class = getattr(__import__(path), module)
            if config is not None:
                self.players[name] = player_class(config)
            else:
                self.players[name] = player_class()

    def run_episode(self):
        obs = self.game_env.reset_game(self.game_config)
        done = False
        self.current_step = 0
        while not done:
            actions = {agent_name: agent.choose_action(obs, self.config["trainning_config"]["use_epsilon"])
                       for agent_name, agent in self.players.items()}
            next_obs, rewards, done, info = self.game_env.update(actions)
            self.replay_buffer.push(obs, actions, rewards, next_obs, done)
            obs = next_obs

            self.current_step += 1
            if self.current_step > self.max_step:
                done = True
        print(self.current_step)

    def train(self, batch_size=32):
        for agent in self.players.values():
            if len(agent.memory) > batch_size:
                agent.train(batch_size)

    def train(self, batch_size=32):
        if len(self.agent1.memory) > batch_size:
            self.agent1.train(batch_size)
        if len(self.agent2.memory) > batch_size:
            self.agent2.train(batch_size)

    def close(self):
        self.env.close()


class DistributedTraining():
    def __init__(self, config):
        self.config = config
        self.num_envs = self.config["trainning_config"]["num_envs"]
        self.num_episodes = self.config["trainning_config"]["num_episodes"]
        self.envs = [DistributedGameEnv.remote(
            config) for _ in range(self.num_envs)]

    def run_training(self):
        for i in range(self.num_episodes):
            results = ray.get(
                [env.run_episode.remote() for env in self.envs])
            for env in self.envs:
                env.train.remote()
            print(f"Episode {i+1}, results: {results}")

    def close_envs(self):
        for env in self.envs:
            env.close.remote()


def main():
    # 环境
    name = 'sea_war'
    weapons_path = 'data/weapons.json'
    scenarios_path = 'data/scenario.json'
    map_path = 'data/map.json'

    scenario = Scenario(scenarios_path, name)
    map = Map(map_path)
    weapon = Weapon(weapons_path)

    game_config = {"name": name,
                   "scenario": scenario,
                   "map": map,
                   "weapon": weapon}

    # 智能体
    AI_agent_config = {
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "learning_rate": 0.001,
        "model": "PPO",
        "state_size": 100,
        "action_size": 50
    }

    # 智能体设置，智能体数量与想定文件scenario一致
    player_config = {
        "red": ("player_AI", "AIPlayer", AI_agent_config),
        "blue": ("rule_agent", "RulePlayer")
    }

    # 分布式训练参数
    trainning_config = {
        "max_step": 1000,
        "num_envs": 4,
        "num_episodes": 100,
        "use_epsilon": True,
        "buffer_capacity": 2000
    }

    config = {
        "game_config": game_config,
        "player_config": player_config,
        "trainning_config": trainning_config
    }
    DistributedGameEnv(config)

    ray.init()
    training = DistributedTraining(config)
    training.run_training()
    training.close_envs()
    ray.shutdown()


if __name__ == '__main__':
    main()
