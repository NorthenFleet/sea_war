import time
import sys
import threading
from env import Env
from player_human import HumanPlayer
from player_AI import AIPlayer
from com_client import CommunicationClient
from render import Render
from init import Map, Weapon, Scenario
from env_tank import EnvTank


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
    def __init__(self, config, net=False):
        self.config = config
        self.game_env = EnvTank(self.config.env_config) if not net else Env(
            name="SC2Env", player_config=config['player_config'])
        self.current_step = None
        self.render = Render()
        self.max_step = self.config.max_step
        self.net = net

        self.players = {}
        for name, (path, module, config) in self.config.player_config.items():
            player_class = getattr(__import__(path), module)
            if config is not None:
                self.players[name] = player_class(config)
            else:
                self.players[name] = player_class()

        if net:
            self.network_client = CommunicationClient(server_host, server_port)
        else:
            self.human_player = HumanPlayer(name="HumanPlayer")

    def start(self):
        if self.net:
            threading.Thread(target=self.network_client.start).start()
        self.run_game()

    def run_game(self):
        observation = self.game_env.reset_game(self.config.env_config)
        game_over = False
        self.current_step = 0
        frame_time = 0.1  # 每帧的时间间隔（秒）
        while not game_over:
            start_time = time.time()
            actions = {}
            if self.net:
                for name, player in self.players.items():
                    actions[name] = player.act(observation)  # 假设所有玩家都是AI
                self.network_client.send_action(actions)
                while self.network_client.received_actions is None:
                    if time.time() - start_time > frame_time:
                        break
                    time.sleep(0.001)
                if self.network_client.received_actions is not None:
                    action_dict = eval(self.network_client.received_actions)
                    self.network_client.received_actions = None
                else:
                    action_dict = {}
            else:
                actions['red'] = self.human_player.choose_action(observation)
                actions['blue'] = self.players['blue'].choose_action(
                    observation)
                action_dict = actions

            observations, rewards, game_over, info = self.game_env.update(
                action_dict)
            self.current_step += 1

            if self.current_step > self.max_step:
                game_over = True

            if self.net and hasattr(self.players['red'], 'remember'):
                self.players['red'].remember(
                    observation, actions['red'], rewards[0], observations, game_over)
                if len(self.players['red'].memory) > 32:
                    self.players['red'].replay(32)

            observation = observations
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)

        print(f"Game Over! Total steps: {self.current_step}")

    def get_human_action(self):
        # Implement method to get human player action
        return 'move_up'  # Example action


# 使用示例
if __name__ == '__main__':
    # 定义全局配置字典
    config = {
        'name': 'battle_royale',
        'max_step': 1000,
        'weapons_path': 'data/weapons.json',
        'scenarios_path': 'data/scenario.json',
        'map_path': 'data/map.json',
        'ai_config': {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "model": "PPO",
            "state_size": 100,
            "action_size": 50,
            "use_epsilon": True,
        },
        'player_config': {
            "red": ("player_AI", "AIPlayer", None),
            "blue": ("player_rule", "RulePlayer", None)
        }
    }

    net = '--net' in sys.argv
    server_host = '127.0.0.1'
    server_port = 9999
    game_config = GameConfig()
    game = Game(game_config, net=net)
    game.start()
