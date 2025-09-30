import ray
import torch.nn as nn
import torch
import torch.optim as optim
from collections import deque
import random
from env import Env
from init import Map, Weapon, Scenario
import numpy as np
from replay_bufer import ReplayBuffer


@ray.remote
class SupervisedGameEnv:
    def __init__(self, config):
        self.config = config
        self.game_env = Env(self.config["game_config"])
        self.memory = deque(maxlen=10000)
        self.model = self.build_model(
            config["state_size"], config["action_size"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = config["trainning_config"]["batch_size"]

    def build_model(self, state_size, action_size):
        """构建神经网络模型"""
        return nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def run_episode(self):
        obs = self.game_env.reset_game(self.config["game_config"])
        done = False
        while not done:
            actions = {agent_name: agent.choose_action(obs)
                       for agent_name, agent in self.players.items()}
            next_obs, rewards, done, info = self.game_env.update(actions)
            for agent_name, reward in rewards.items():
                self.remember(obs, actions[agent_name], reward)
            obs = next_obs

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards = zip(*minibatch)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)

        pred_actions = self.model(states)
        action_one_hot = torch.nn.functional.one_hot(
            actions, num_classes=self.config["action_size"])
        pred_values = torch.sum(pred_actions * action_one_hot, dim=1)

        loss = nn.MSELoss()(pred_values, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Training loss: {loss.item()}")

    def close(self):
        self.game_env.close()


@ray.remote
class DistributedGameEnv:
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.max_step = config["trainning_config"]["max_step"]
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
        obs = self.game_env.reset_game(self.config["game_config"])
        done = False
        self.current_step = 0
        while not done:
            actions = {agent_name: agent.choose_action(obs)
                       for agent_name, agent in self.players.items()}
            next_obs, rewards, done, info = self.game_env.update(actions)
            # ToDo 不能把所有动作放到一个四元组中，每个智能体存放自己的记录，然后合成到一起。
            self.replay_buffer.push(obs, actions, rewards, next_obs, done)
            obs = next_obs

            self.current_step += 1
            if self.current_step > self.max_step:
                done = True

        for agent_name, agent in self.players.items():
            self.replay_buffer.push(agent.memory)

        print(self.current_step)

    def train(self, batch_size=32, alpha=0.6):
        if len(self.replay_buffer) > batch_size:
            samples, indices = self.replay_buffer.sample(batch_size, alpha)
            for agent in self.players.values():
                agent.train(samples)
            priorities = [max(abs(sample[2]), 1.0) for sample in samples]
            self.replay_buffer.update_priorities(indices, priorities)

    def close(self):
        self.env.close()


class DistributedTraining():
    def __init__(self, config):
        self.config = config
        self.num_envs = self.config["trainning_config"]["num_envs"]
        self.num_episodes = self.config["trainning_config"]["num_episodes"]
        self.envs = [DistributedGameEnv.remote(
            config) for _ in range(self.num_envs)]

    def supervised_training(self):
        for i in range(self.config["trainning_config"]["num_episodes"]):
            ray.get([env.run_episode.remote() for env in self.envs])
            ray.get([env.train.remote() for env in self.envs])
            print(f"Episode {i+1} completed")

    def rl_training(self):
        for i in range(self.num_episodes):
            results = ray.get(
                [env.run_episode.remote() for env in self.envs])
            for env in self.envs:
                env.train.remote()
            print(f"Episode {i+1}, results: {results}")

    def close_envs(self):
        for env in self.envs:
            env.close.remote()


if __name__ == '__main__':
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

    # 智能体设置，智能体数量与想定文件scenario一致
    player_config = {
        "red": ("player_AI", "AIPlayer", agent_config),
        "blue": ("player_rule", "RulePlayer", None)
    }

    # 分布式训练参数
    trainning_config = {
        "max_step": 1000,
        "num_envs": 2,
        "num_episodes": 100,
        "buffer_capacity": 2000
    }

    config = {
        "game_config": game_config,
        "player_config": player_config,
        "trainning_config": trainning_config
    }

    ray.init()
    training = DistributedTraining(config)
    training.rl_training()
    training.close_envs()
    ray.shutdown()
