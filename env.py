from gym import spaces
import gym
import numpy as np


class GameEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, name, agent_modules):
        super(GameEnv, self).__init__()
        # 动态导入智能体模块
        # self.agents = {name: getattr(__import__(module), cls)(
        #     name) for name, (module, cls) in agent_modules.items()}

        self.agents = {}
        for name, (module, cls, training_config, model) in agent_modules.items():
            agent_class = getattr(__import__(module), cls)
            if model is not None:
                self.agents[name] = agent_class(name, training_config, model)
            else:
                self.agents[name] = agent_class(name)

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        self.name = name
        self.Scenario = None
        self.map = None
        self.weapon = None
        self.state = None

        self.current_step = None

    def reset_game(self, config):
        # 重置游戏
        self.state = config["scenario"]
        self.map = config["map"]
        self.weapon = config["weapon"]
        self.current_step = 0
        self.game_over = False
        print("Game starts with the following units:")

        return {name: self.observation_space.sample() for name in self.agents}

    def update(self, action_dict):
        # 解析动作字典，执行动作
        rewards = {}
        for agent_name, action in action_dict.items():
            # 这里的奖励函数非常简单：如果动作是1，奖励为1，否则为0
            rewards[agent_name] = 1 if action == 1 else 0

        # 简单示例：观测值为随机
        observations = {name: self.observation_space.sample()
                        for name in self.agents}
        done = False
        return observations, rewards, done, {}
