from .sea_war_env import SeaWarEnv
from render.single_process import RenderManager
from ui.start_menu import StartMenu
from ui.player_human import HumanPlayer
from ai.player_rule import RulePlayer
from ai.player_rule_blue import BluePlayer
from ai.player_rule_red import RedPlayer
from .event_manager import EventManager
from .communication import CommunicationClient, CommunicationServer
import time
import threading


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
        if self.render_manager is None:
            self.render_manager = RenderManager(self.env, self.screen_size, terrain_override=terrain_override)

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


# 使用示例
if __name__ == '__main__':
    # 定义全局配置字典

    game_config = {
        'name': 'AirDefense',
        'device_path': 'core/data/device_new.json',
        'scenario_path': 'core/data/skirmish_1.json',
        'map_path': 'core/data/map.json',
    }

    players = {
        "red": 'Red',
        "blue": 'Blue'
        # "blue": ("RulePlayer", HumanPlayer),
        # "green": ("HumanPlayer", RulePlayer)
    }

    game = Game(game_config, players)
    # 先展示启动菜单选择地图
    menu = StartMenu()
    selected_map = menu.run(screen_size=(1280, 800), auto_select_timeout=1.0)  # 返回文件名或 None
    try:
        game.run(terrain_override=selected_map)
    except KeyboardInterrupt:
        print('用户中断，正在退出...')
    finally:
        # 优雅停止通信线程
        if hasattr(game, 'communication_client'):
            try:
                game.communication_client.stop()
            except Exception:
                pass
        if hasattr(game, 'communication_server'):
            try:
                game.communication_server.stop()
            except Exception:
                pass
        print('已退出')
