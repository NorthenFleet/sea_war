from sea_war_env import SeaWarEnv
from render.single_process import RenderManager
from player_AI import AIPlayer
from player_human import HumanPlayer
from player_rule import RulePlayer
from player_rule_blue import BluePlayer
from player_rule_red import RedPlayer
from event_manager import EventManager
from communication import CommunicationClient, CommunicationServer
import time
import threading


class Game:
    def __init__(self, game_config, players, is_server=False):
        self.env = SeaWarEnv(game_config)
        self.current_time = 0.0
        self.fixed_time_step = 1 / 60  # 固定时间步长
        screen_size = (1000, 1200)
        self.render_manager = RenderManager(self.env, screen_size)
        self.players = {}

        # 注册系统事件
        self.event_manager = EventManager()
        self.event_manager.subscribe('GameOver', self.handle_game_over)

        # 注册 AI 玩家
        for player, player_type in players.items():
            if player_type == 'AI':
                self.players[player] = AIPlayer()
            elif player_type == 'Human':
                self.players[player] = HumanPlayer()
            elif player_type == 'Rule':
                self.players[player] = RulePlayer(player_type)
            elif player_type == 'Red':
                self.players[player] = RedPlayer(player_type)
            elif player_type == 'Blue':
                self.players[player] = BluePlayer(
                    player_type, self.env.device_table)
            else:
                raise ValueError(
                    f'Invalid player type: {player_type}')

        # 初始化通信系统
        if is_server:
            self.communication_server = CommunicationServer()
            self.communication_server.start()
        else:
            self.communication_client = CommunicationClient(
                'server_host', 9999)
            threading.Thread(target=self.communication_client.start).start()

    def run(self):
        game_data, sides = self.env.reset_game()
        game_over = False
        self.current_step = 0

        while not game_over:
            start_time = time.time()
            # 1. 渲染
            self.render_manager.update()

            # 2. 处理玩家动作
            actions = []
            for player, agent in self.players.items():
                action = agent.choose_action(sides[player])
                actions.append(action)

            # 3. 处理网络数据
            netactions = self.network_update()
            actions.append(netactions)

            # 4. 更新游戏状态
            self.env.update(actions, self.fixed_time_step)

            # 5. 处理游戏结束
            if self.env.game_over:
                # self.event_manager.post(Event('GameOver', {}))
                print('Game Over')
                break

            # 控制帧率
            self.limit_fps(start_time)

    def handle_game_over(self):
        pass

    def network_update(self):
        self.env.network_update()

    def limit_fps(self, start_time):
        # 控制帧率，例如每秒 60 帧
        frame_duration = time.time() - start_time
        time_to_sleep = max(0, (1 / 60) - frame_duration)
        time.sleep(time_to_sleep)


# 使用示例
if __name__ == '__main__':
    # 定义全局配置字典

    game_config = {
        'name': 'AirDefense',
        'device_path': 'data/device_new.json',
        'scenario_path': 'data/air_defense.json',
        'map_path': 'data/map.json',
    }

    players = {
        "red": 'Red',
        "blue": 'Blue'
        # "blue": ("RulePlayer", HumanPlayer),
        # "green": ("HumanPlayer", RulePlayer)
    }

    game = Game(game_config, players)
    game.run()
