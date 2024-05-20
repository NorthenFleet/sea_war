from gym import spaces
import numpy as np


\
class Env():
    metadata = {'render.modes': ['human']}

    def __init__(self, game_config, com=None):
        super(Env, self).__init__()
        self.name = game_config["name"]
        self.scenario = game_config["scenario"]
        self.map = game_config["map"]
        self.weapon = game_config["weapon"]
        self.state = game_config["scenario"]
        self.current_step = None
        self.com = com

        self.players = self.scenario.players
        
        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

    def reset_game(self, config):
        # 重置游戏
        self.current_step = 0
        self.game_over = False
        print("Game starts with the following units:")

        return {name: self.observation_space.sample() for name in self.players}

    def update(self, action_dict):
        # 解析动作字典，执行动作
        rewards = {}
        for agent_name, action in action_dict.items():
            # 这里的奖励函数非常简单：如果动作是1，奖励为1，否则为0
            rewards[agent_name] = 1 if action == 1 else 0

        # 简单示例：观测值为随机
        observations = {name: self.observation_space.sample()
                        for name in self.players}
        done = False
        # np.reshape(
        #     observations, [1, self.input_dim])
        return observations, rewards, done, {}
