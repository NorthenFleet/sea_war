from .sea_war_env import SeaWarEnv
from ..render.single_process import RenderManager
from ..ui.start_menu import StartMenu
from ..ui.player_human import HumanPlayer
from ..ai.player_rule import RulePlayer
from ..ai.player_rule_blue import BluePlayer
from ..ai.player_rule_red import RedPlayer
from .event_manager import EventManager
from .communication import CommunicationClient, CommunicationServer
import time
import threading
import argparse


class Game:
    def __init__(self, game_config, players, is_server=False):
        self.env = SeaWarEnv(game_config)
        self.current_time = 0.0
        self.fixed_time_step = 1 / 60  # 固定时间步长
        self.paused = False
        self.speed_scale = 1.0
        # 横屏窗口尺寸
        self.screen_size = (1280, 800)
        self.render_manager = None
        self.players = {}

        # 注册系统事件
        self.event_manager = EventManager()
        self.event_manager.subscribe('GameOver', self.handle_game_over)

        # 注册 AI 玩家
        for player, player_type in players.items():
            if player_type == 'AI':
                from ai.player_AI import AIPlayer
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
            threading.Thread(target=self.communication_client.start, daemon=True).start()

    def run(self, terrain_override=None):
        game_data, sides = self.env.reset_game()
        game_over = False
        self.current_step = 0

        # 在运行前创建渲染器（携带地图选择）
        # 优先使用传入的地形覆盖，其次使用环境绑定的默认地图图片
        effective_terrain = terrain_override
        if effective_terrain is None and hasattr(self.env, 'default_map_image') and self.env.default_map_image:
            effective_terrain = self.env.default_map_image
            print(f'使用场景绑定的地图图片: {effective_terrain}')
        
        if self.render_manager is None:
            # 若使用图片地图且未绑定 map_json，则关闭障碍绿色方块叠加
            show_obstacles = bool(getattr(self.env, 'default_map_json', None))
            self.render_manager = RenderManager(self.env, self.screen_size, terrain_override=effective_terrain, show_obstacles=show_obstacles)

        while not game_over:
            start_time = time.time()
            # 1. 渲染（返回 False 表示用户请求关闭窗口）
            if not self.render_manager.update():
                break

            # 2. 处理玩家动作
            all_command_list = []
            for player, agent in self.players.items():
                action = agent.choose_action(sides[player])
                if action is not None:
                    all_command_list.append(action)

            # 2.1 处理前端鼠标指令
            human_commands = self.render_manager.consume_commands()
            if human_commands and len(human_commands.get_commands()) > 0:
                all_command_list.append(human_commands)

            # 2.2 处理UI动作（暂停/加速/减速）
            for act in self.render_manager.consume_ui_actions():
                if act == 'pause_toggle':
                    self.paused = not self.paused
                elif act == 'speed_up':
                    self.speed_scale = min(4.0, self.speed_scale + 0.25)
                elif act == 'speed_down':
                    self.speed_scale = max(0.25, self.speed_scale - 0.25)

            # 3. 处理网络玩家动作
            netactions = self.network_update()
            if netactions is not None:
                all_command_list.append(netactions)

            # # 4. 处理玩家指令
            self.env.process_commands(all_command_list)

            # 5. 更新游戏状态
            if not self.paused:
                self.env.update(self.fixed_time_step * self.speed_scale)

            # 6. 处理游戏结束
            if self.env.game_over:
                # self.event_manager.post(Event('GameOver', {}))
                print('Game Over')
                break

            # 控制帧率
            if not self.limit_fps(start_time):
                # 用户中断（Ctrl+C）时优雅退出
                break

    def handle_game_over(self):
        pass

    def network_update(self):
        self.env.network_update()

    def limit_fps(self, start_time):
        # 控制帧率，例如每秒 60 帧
        frame_duration = time.time() - start_time
        time_to_sleep = max(0, (1 / 60) - frame_duration)
        try:
            time.sleep(time_to_sleep)
        except KeyboardInterrupt:
            # 捕获中断以避免非零退出码
            return False
        return True
