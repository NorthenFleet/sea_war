from sea_war_env import SeaWarEnv
from render.render_manager import RenderManager
from player_AI import AIPlayer
from player_human import HumanPlayer
from player_rule import RulePlayer
from event_manager import EventManager
from component_manager import *
from entities.aircraft import Aircraft
from communication import CommunicationClient, CommunicationServer
from game_data import GameData
import time
import threading


class Game:
    def __init__(self, game_config, players, is_server=False):
        self.env = SeaWarEnv(game_config)
        self.game_data = GameData()
        self.players = players
        self.event_manager = EventManager()
        self.entities = []
        self.systems = []
        self.current_time = 0.0
        self.fixed_time_step = 1 / 60  # 固定时间步长

        # self.init_entities()
        self.init_systems()

        # 初始化通信系统
        if is_server:
            self.communication_server = CommunicationServer()
            self.communication_server.start()
        else:
            self.communication_client = CommunicationClient(
                'server_host', 9999)
            threading.Thread(target=self.communication_client.start).start()

    def init_entities(self):
        # Initialize entities and add to the list
        f18 = Aircraft(self.event_manager)
        f18.initialize(position=(0, 0), health=5000,
                       movement_speed=10, weapon_damage=50, weapon_range=100)
        self.entities.append(f18)

    def init_systems(self):
        # Initialize systems and add entities to them
        self.movement_system = MovementSystem(self.game_data)
        self.attack_system = AttackSystem(self.game_data)
        self.dot_system = DamageOverTimeSystem(self.game_data)

        for entity in self.entities:
            self.movement_system.add_entity(entity)
            self.attack_system.add_entity(entity)
            self.dot_system.add_entity(entity)

        self.systems.extend(
            [self.movement_system, self.attack_system, self.dot_system])

    def run(self):
        game_data, sides = self.env.reset_game()
        game_over = False
        self.current_step = 0

        while not game_over:
            start_time = time.time()

            # 1. 处理输入事件
            self.event_manager.post(Event('FrameStart', {}))

            # 2. 逻辑更新
            self.current_time += self.fixed_time_step
            self.update_logic(self.fixed_time_step)

            # 3. 渲染
            self.render_manager.update()

            # 4. 网络处理
            self.network_update()

            # 5. 玩家动作选择并执行
            actions = []
            for player, agent in self.players.items():
                action = agent.choose_action(sides[player])
                actions.append(action)

            # 6. 更新游戏环境
            observations, rewards, game_over, info = self.env.update(actions)

            # 7. 处理游戏结束
            if game_over:
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

    def update_logic(self, delta_time):
        # Update all systems and entities
        for system in self.systems:
            system.update(delta_time)

        # Update all entities' state machines
        for entity in self.entities:
            entity.update(delta_time)

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
