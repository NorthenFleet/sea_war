from render import Render
from init import Map, Weapon, Scenario
from env_tank import EnvTank
import json


class GameConfig:
    def __init__(self):
        # 从全局配置字典中加载参数
        self.name = config['name']
        self.max_step = config['max_step']
        self.weapons_path = config['weapons_path']
        self.scenarios_path = config['scenarios_path']
        self.map_path = config['map_path']

        self.scenario = Scenario(self.scenarios_path, self.name)
        self.map = Map(self.map_path)
        self.weapon = Weapon(self.weapons_path)

        self.env_config = {
            "name": self.name,
            "scenario": self.scenario,
            "map": self.map,
            "weapon": self.weapon
        }

        self.ai_config = config['ai_config']
        self.player_config = config['player_config']

        # 更新全局配置字典中的玩家配置
        for name, (path, module, _) in self.player_config.items():
            if module == "AIPlayer":
                config['player_config'][name] = (path, module, self.ai_config)


class Game:
    def __init__(self, config):
        self.config = config
        self.game_env = EnvTank(self.config.env_config)
        self.current_step = None
        self.render = Render()
        self.max_step = self.config.max_step

        self.players = {}
        for name, (path, module, config) in self.config.player_config.items():
            player_class = getattr(__import__(path), module)
            if config is not None:
                self.players[name] = player_class(config)
            else:
                self.players[name] = player_class()

    def run(self):
        observation = self.game_env.reset_game(self.config.env_config)
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = {agent_name: agent.choose_action(
                observation) for agent_name, agent in self.players.items()}
            observations, rewards, game_over, info = self.game_env.update(
                actions)

            self.current_step += 1
            print(self.current_step)

            if self.current_step > self.max_step:
                game_over = True


# 使用示例
if __name__ == '__main__':
    # 定义全局配置字典

    game_config = {
        'name': 'battle_royale',

        'weapons_path': 'data/weapons.json',
        'scenarios_path': 'data/scenario.json',
        'map_path': 'data/map.json',
    }
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

    player_config = {
        "red": ("player_AI", "AIPlayer", agent_config),
        "blue": ("player_rule", "RulePlayer", None)
    }

    trainning_config = {
        'max_step': 1000,
    }

    config = {


    }

    config = GameConfig()
    game = Game(config)
    game.run()
