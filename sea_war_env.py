import numpy as np
from gym import spaces
from env import Env
from game_data import GameData
from init import Map, Device, Side, Scenario
from entities.entity import EntityInfo
from init import Grid, QuadTree
from utils import *
from component_manager import *
from event_manager import EventManager
from component_manager import Event
# 定义为游戏的战术层，从战术层面对游戏过程进行解析


class SeaWarEnv(Env):
    def __init__(self, game_config):
        self.name = game_config["name"]
        self.map = None
        self.device_table = None
        self.scenario = None

        self.sides = {}
        self.actions = {}
        self.game_over = False
        self.current_step = 0

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        self.map = Map(game_config['map_path'])
        self.device_table = Device(game_config['device_path'])
        self.scenario = Scenario(game_config['scenario_path'])

        self.grid = Grid(1000, 100)
        self.quad_tree = QuadTree([0, 0, 1000, 1000], 4)

        # 游戏逻辑事件系统
        self.event_manager = EventManager()

        self.game_data = GameData(self.event_manager)

        # 初始化系统
        self.movement_system = MovementSystem(self.event_manager)
        self.attack_system = AttackSystem(self.event_manager)
        self.detection_system = DetectionSystem(
            self.event_manager, self.quad_tree, self.grid)

        self.pathfinding_system = PathfindingSystem(
            self.event_manager, self.map)

        # 组件系统注册到一个列表中，便于统一管理
        self.systems = [
            self.movement_system,
            self.attack_system,
            self.detection_system,
            self.pathfinding_system
        ]

    def load_scenario(self, scenario):
        """
        从想定文件中加载场景，并初始化 ECS 实体和组件系统。
        """
        for color, units in scenario.data.items():
            for unit_id, unit_info in units.items():
                # 根据想定文件构造 EntityInfo
                entity_info = EntityInfo(
                    entity_id=unit_info["id"],
                    side=unit_info["side"],
                    position=[unit_info["x"], unit_info["y"], unit_info["z"]],
                    rcs=unit_info["rcs"],
                    entity_type=unit_info["entity_type"],
                    heading=unit_info["heading"],
                    speed=unit_info["speed"],
                    health=unit_info["health"],
                    endurance=unit_info["endurance"],
                    weapons=unit_info["weapons"],
                    sensor=unit_info["sensor"]
                )
                self.game_data.add_entity(entity_info, None, color)
            # Create the player side (e.g., for blue/red teams)
            side = Side(color)
            side.set_entities(self.game_data.get_player_unit_ids(color))
            self.sides[color] = side

        return self.game_data

    def load_scenario_OOP(self, scenario):
        """
        Load the scenario into GameData and return the updated GameData object.
        """
        game_data = GameData(
            self.game_data)  # Initialize a new GameData instance

        for color, unit_list in scenario.data.items():
            for unitid, unit in unit_list.items():
                entity_info = EntityInfo(
                    entity_id=unit['id'],
                    entity_type=unit['entity_type'],
                    position={"x": unit['x'], "y": unit['y'], "z": unit['z']},
                    rcs=unit['rcs'],
                    speed=unit['speed'],
                    direction=unit['course'],
                    hp=unit['health'],
                    weapons=[w['type'] for w in unit['weapons']],
                    sensor=[s['type'] for s in unit['sensor']]
                )
                # Add entity to game_data
                # device=None for now
                game_data.add_entity(entity_info, None, color)

            # Create the player side (e.g., for blue/red teams)
            side = Side(color)
            side.set_entities(game_data.get_player_unit_ids(color))
            self.sides[color] = side

        # Return the populated game_data object
        return game_data

    def game_data_to_component(self, game_data):
        """
        转换 GameData 到 ComponentManager
        """
        # 转换所有实体到 ComponentManager
        for entity_id, entity in game_data.entities.items():
            position = PositionComponent(
                entity_id, entity.position['x'], entity.position['y'], entity.position['z'])
            detection = DetectionComponent(
                entity_id, entity.rcs, entity.sensor)
            movement = MovementComponent(
                entity_id, entity.speed, entity.direction)
            attack = AttackComponent(entity_id, entity.weapons)
            self.game_data.add_component(entity_id, position)
            self.game_data.add_component(entity_id, detection)
            self.game_data.add_component(entity_id, movement)

    def reset_game(self):
        self.current_step = 0
        self.game_over = False
        self.game_data.reset()
        # Load the scenario and update GameData
        self.game_data = self.load_scenario(self.scenario)
        return self.game_data, self.sides

    def update(self, actions):
        # 处理玩家动作
        for action in actions:
            if action == 'move':
                self.movement_system.update(self.game_data, 1 / 60)
            elif action == 'attack':
                self.attack_system.update(self.game_data)

        # 调用各个系统更新游戏状态
        self.pathfinding_system.update(1 / 60)
        self.detection_system.update(1 / 60)
        self.collision_system.update(1 / 60)
        # Update all systems and entities
        for system in self.systems:
            system.update(delta_time)

        # Update all entities' state machines
        for entity in self.entities:
            entity.update(delta_time)

        self.current_step += 1

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
