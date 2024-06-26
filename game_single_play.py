from init import Initializer
from env_tank import EnvTank
from render.render_manager import RenderManager


class GameData:
    def __init__(self, config):
        # 从全局配置字典中加载参数
        self.name = config['name']
        # self.max_step = config['max_step']
        self.weapons_path = config['weapons_path']
        self.scenarios_path = config['scenarios_path']
        self.map_path = config['map_path']

        # self.scenario = Scenario(self.scenarios_path, self.name)
        # self.map = Map(self.map_path)
        # self.weapon = Weapon(self.weapons_path)

        # self.env_config = {
        #     "name": self.name,
        #     "scenario": self.scenario,
        #     "map": self.map,
        #     "weapon": self.weapon
        # }

        # self.ai_config = config['ai_config']
        # self.player_config = config['player_config']

        # # 更新全局配置字典中的玩家配置
        # for name, (path, module, _) in self.player_config.items():
        #     if module == "AIPlayer":
        #         config['player_config'][name] = (path, module, self.ai_config)


class Game:
    def __init__(self, game_config, agent_config, player_config, trainning_config=None):
        initializer = Initializer(game_config)
        self.env_config = initializer.get_env_config()
        self.game_env = EnvTank(self.env_config)

        self.players = {}
        for name, (path, module, config) in player_config.items():
            player_class = getattr(__import__(path), module)
            if config is not None:
                self.players[name] = player_class(agent_config)
            else:
                self.players[name] = player_class()

        self.current_step = None
        self.render_manager = RenderManager(self.env_config)

    def run(self):
        observation = self.game_env.reset_game()
        self.render_manager.run()
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = {agent_name: agent.choose_action(
                observation) for agent_name, agent in self.players.items()}
            observations, rewards, game_over, info = self.game_env.update(
                actions)

            self.current_step += 1
            print(self.current_step)

            if game_over:
                break

    def train(self):
        observation = self.game_env.reset_game()
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = {agent_name: agent.choose_action(
                observation) for agent_name, agent in self.players.items()}
            observations, rewards, game_over, info = self.game_env.update(
                actions)

            self.current_step += 1
            print(self.current_step)

            if game_over:
                break


# 使用示例
if __name__ == '__main__':
    # 定义全局配置字典

    game_config = {
        'name': 'AirDefense',
        'device_path': 'data/device.json',
        'scenarios_path': 'data/AirDefense.json',
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
    rule_config = {
        "name": "Rule"
    }
    human_config = {
        "name": "Human"
    }

    player_config = {
        "red": ("player_AI", "AIPlayer", agent_config),
        "blue": ("player_rule", "RulePlayer", rule_config),
        "green": ("player_human", "HumanPlayer", human_config)
    }

    trainning_config = {
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "learning_rate": 0.001,
        "model": "PPO",
        "state_size": 100,
        "action_size": 50,
        "use_epsilon": True,
        'training_max_step': 1000,
    }

    # config = {
    #     'name': game_config['name'],
    #     'player_config': player_config,
    #     'ai_config': agent_config,
    #     'env_config': game_config,
    #     'trainning_config': trainning_config
    # }

    # config = GameData(game_config)
    game = Game(game_config, agent_config, player_config)
    game.run()
    # game.train()
