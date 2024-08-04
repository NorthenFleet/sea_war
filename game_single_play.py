from init import Initializer
from sea_war_env import SeaWarEnv
from render.render_manager import RenderManager
from player_AI import AIPlayer
from player_human import HumanPlayer
from player_rule import RulePlayer


class Game:
    def __init__(self, game_config, players):
        self.env = SeaWarEnv(game_config)
        self.players = players

        # 动态载入player类
        # for name, (path, module, config) in player_config.items():
        #     player_class = getattr(__import__(path), module)
        #     if config is not None:
        #         self.players[name] = player_class(agent_config)
        #     else:
        #         self.players[name] = player_class()

        self.current_step = None
        self.render_manager = RenderManager(self.env_info)

    def run(self):
        observation = self.env.reset_game()
        # self.render_manager.run()
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = []
            for player, agent in self.players.items():
                action = agent.choose_action(observation)
                actions.append(action)

            # actions = {agent_name: agent.choose_action(
            #     observation) for agent_name, agent in self.players.items()}
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
        'scenario_path': 'data/AirDefense.json',
        'map_path': 'data/map.json',
    }

    red_player = RulePlayer("red")
    blue_player = RulePlayer("blue")

    player = {
        "red": red_player,
        "blue": blue_player
        # "blue": ("RulePlayer", HumanPlayer),
        # "green": ("HumanPlayer", RulePlayer)
    }

    game = Game(game_config, player)
    game.run()
