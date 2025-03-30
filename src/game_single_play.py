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
import json
import os


class Game:
    def __init__(self, game_config, players, is_server=False, server_host='localhost', server_port=9999):
        self.env = SeaWarEnv(game_config)
        self.current_time = 0.0
        self.fixed_time_step = 1 / 60  # 固定时间步长
        screen_size = (1000, 1200)
        self.render_manager = RenderManager(self.env, screen_size)
        self.players = {}
        self.is_server = is_server
        self.game_over = False
        
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
            self.communication_server = CommunicationServer(port=server_port)
            self.communication_server.start()
            # 服务器模式下启动UE通信服务
            # self.ue_communication = UECommunicationServer(port=9998)
            self.ue_communication.start()
        else:
            self.communication_client = CommunicationClient(
                server_host, server_port)
            threading.Thread(target=self.communication_client.start, daemon=True).start()
            # 客户端模式下启动UE通信客户端
            # self.ue_communication = UECommunicationClient(port=9998)
            # threading.Thread(target=self.ue_communication.start, daemon=True).start()

    def run(self):
        game_data, sides = self.env.reset_game()
        self.game_over = False
        self.current_step = 0

        while not self.game_over:
            start_time = time.time()
            # 1. 渲染
            self.render_manager.update()

            # 2. 处理玩家动作
            all_command_list = []
            for player, agent in self.players.items():
                action = agent.choose_action(sides[player])
                if action is not None:
                    all_command_list.append(action)

            # 3. 处理网络玩家动作
            # net_actions = self.network_update()
            # if net_actions:
            #     all_command_list.extend(net_actions)
                
            # 4. 处理UE输入的指令
            ue_commands = self.get_ue_commands()
            if ue_commands:
                all_command_list.extend(ue_commands)

            # 5. 处理玩家指令
            self.env.process_commands(all_command_list)

            # 6. 更新游戏状态
            self.env.update(self.fixed_time_step)
            
            # 7. 发送游戏状态到UE
            self.send_game_state_to_ue()

            # 8. 如果是服务器，广播游戏状态到所有客户端
            if self.is_server:
                self.broadcast_game_state()

            # 9. 处理游戏结束
            if self.env.game_over:
                self.event_manager.post_event('GameOver', {})
                self.game_over = True
                print('Game Over')
                break

            # 控制帧率
            self.limit_fps(start_time)
            self.current_step += 1

    def handle_game_over(self, event_data):
        self.game_over = True
        # 通知UE游戏结束
        if hasattr(self, 'ue_communication'):
            self.ue_communication.send_message(json.dumps({
                'type': 'game_over',
                'data': event_data
            }))

    def network_update(self):
        """处理网络更新，获取其他客户端的命令"""
        if self.is_server:
            # 服务器从所有客户端收集命令
            return self.communication_server.get_all_commands()
        else:
            # 客户端从服务器获取其他客户端的命令
            return self.communication_client.get_commands()

    def broadcast_game_state(self):
        """服务器广播游戏状态到所有客户端"""
        if not self.is_server:
            return
            
        game_state = self.env.get_serialized_state()
        self.communication_server.broadcast_message(json.dumps({
            'type': 'game_state',
            'step': self.current_step,
            'data': game_state
        }))

    def get_ue_commands(self):
        """获取从UE发送的命令"""
        if hasattr(self, 'ue_communication'):
            return self.ue_communication.get_commands()
        return []

    def send_game_state_to_ue(self):
        """发送游戏状态到UE进行显示"""
        if hasattr(self, 'ue_communication'):
            game_state = self.env.get_serialized_state()
            self.ue_communication.send_message(json.dumps({
                'type': 'game_state',
                'step': self.current_step,
                'data': game_state
            }))

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
        'scenario_path': 'data/air_defense_1.json',
        'map_path': 'data/map.json',
    }

    players = {
        "red": 'Red',
        "blue": 'Blue'
    }

    # 根据命令行参数决定是服务器还是客户端
    import sys
    is_server = len(sys.argv) > 1 and sys.argv[1] == 'server'
    server_host = 'localhost'
    if len(sys.argv) > 2:
        server_host = sys.argv[2]
        
    game = Game(game_config, players, is_server=is_server, server_host=server_host)
    game.run()
