from init import Initializer
from sea_war_env import SeaWarEnv
from render.render_manager import RenderManager
from player_AI import AIPlayer
from player_human import HumanPlayer
from player_rule import RulePlayer

class Game:
    def __init__(self, game_config,  player):
        initializer = Initializer(game_config)
        self.env_config = initializer.get_env_config()
        self.env = SeaWarEnv(self.env_config)

        self.players = player

        # 动态载入player类
        # for name, (path, module, config) in player_config.items():
        #     player_class = getattr(__import__(path), module)
        #     if config is not None:
        #         self.players[name] = player_class(agent_config)
        #     else:
        #         self.players[name] = player_class()

        self.current_step = None
        self.render_manager = RenderManager(self.env_config)

    def run(self):
        observation = self.env.reset_game()
        # self.render_manager.run()
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = {agent_name: agent.choose_action(
                observation) for agent_name, agent in self.players.items()}
            observations, rewards, game_over, info = self.env.update(
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

    player = {
        "red": ("AIPlayer", AIPlayer),
        "blue": ("RulePlayer", AIPlayer)
        # "blue": ("RulePlayer", HumanPlayer),
        # "green": ("HumanPlayer", RulePlayer)
    }

    game = Game(game_config, player)
    game.run()
