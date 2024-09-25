from sea_war_env import SeaWarEnv
from render.render_manager import RenderManager
from player_AI import AIPlayer
from player_human import HumanPlayer
from player_rule import RulePlayer
from event_manager import EventManager
from component_manager import Event
from communication import CommunicationClient, CommunicationServer
from game_data import GameData
import time
import threading


class Game:
    def __init__(self, game_config, players, is_server=False):
        self.event_manager = EventManager()
        self.env = SeaWarEnv(game_config)
        self.players = players
        self.current_time = 0.0
        self.fixed_time_step = 1 / 60  # 固定时间步长
        self.render_manager = RenderManager(self.env)

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

            # 1. 处理输入事件
            self.event_manager.post(Event('FrameStart', {}))

            # 2. 处理玩家动作
            actions = []
            for player, agent in self.players.items():
                action = agent.choose_action(sides[player])
                actions.append(action)

            # 3. 更新游戏状态
            self.env.update(actions, self.fixed_time_step)

            # 4. 渲染
            self.render_manager.update()

            # 5. 网络处理
            self.network_update()

            # 6. 处理游戏结束
            if self.env.game_over:
                self.event_manager.post(Event('GameOver', {}))
                break

            # 控制帧率
            self.limit_fps(start_time)

    def network_update(self):
        if hasattr(self, 'communication_server'):
            # Server-specific logic, e.g., broadcasting collected actions
            pass
        if hasattr(self, 'communication_client'):
            # Client-specific logic, e.g., sending and receiving actions
            if self.communication_client.received_actions:
                # 处理接收到的动作，例如通过事件系统传递给其他系统
                self.event_manager.post(
                    Event('NetworkActionsReceived', self.communication_client.received_actions))

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
        'device_path': 'data/device.json',
        'scenario_path': 'data/AirDefense.json',
        'map_path': 'data/map.json',
    }

    red_player = RulePlayer("red")
    blue_player = RulePlayer("blue")

    players = {
        "red": red_player,
        "blue": blue_player
        # "blue": ("RulePlayer", HumanPlayer),
        # "green": ("HumanPlayer", RulePlayer)
    }

    game = Game(game_config, players)
    game.run()
