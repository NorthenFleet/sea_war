import numpy as np
from gym import spaces
from env import Env
from game_data import GameData
from init import Map, DeviceTableDict, Side, Scenario
from entities.entity import *
from init import Grid, QuadTree
from utils import *
from system_manager import *
from event_manager import EventManager
# 定义为游戏的战术层，从战术层面对游戏过程进行解析


class SeaWarEnv(Env):
    def __init__(self, game_config):
        self.name = game_config["name"]
        self.map = None
        self.scenario = None

        self.sides = {}
        self.actions = {}
        self.game_over = False
        self.current_step = 0

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        self.map = Map(game_config['map_path'])
        self.game_map = self.map_process(self.map, (1000, 1000))  # 地图扩充
        self.device_table = DeviceTableDict(game_config['device_path'])
        self.scenario = Scenario(game_config['scenario_path'])

        self.grid = Grid(1000, 100)
        self.quad_tree = QuadTree([0, 0, 1000, 1000], 4)

        # 游戏逻辑事件系统
        self.event_manager = EventManager()

        self.game_data = GameData(self.event_manager)

        # 初始化系统
        self.movement_system = MovementSystem(
            self.game_data, self.event_manager)
        self.attack_system = AttackSystem(self.game_data, self.event_manager)
        self.detection_system = DetectionSystem(self.game_data,
                                                self.event_manager, self.device_table, self.quad_tree, self.grid)

        self.pathfinding_system = PathfindingSystem(self.game_data,
                                                    self.event_manager, self.game_map)

        # 组件系统注册到一个列表中，便于统一管理
        self.systems = [
            self.movement_system,
            self.attack_system,
            self.detection_system,
            self.pathfinding_system
        ]

    def map_process(self, original_map, target_size):
        original_map = np.array(original_map)  # 转换为NumPy数组便于扩展操作
        original_height, original_width = original_map.shape
        expanded_map = np.tile(
            original_map, (target_size[0] // original_height, target_size[1] // original_width))

        # 如果目标大小不是原始大小的倍数，处理剩余部分
        remaining_height = target_size[0] % original_height
        remaining_width = target_size[1] % original_width

        if remaining_height > 0:
            expanded_map = np.vstack(
                (expanded_map, original_map[:remaining_height, :]))

        if remaining_width > 0:
            expanded_map = np.hstack(
                (expanded_map, expanded_map[:, :remaining_width]))

        return expanded_map.tolist()  # 返回扩展后的地图数据

    def load_scenario(self, scenario):
        """
        从想定文件中加载场景，并初始化 ECS 实体和组件系统。
        """
        for color, units in scenario.data.items():
            entities = []
            for unit_id, unit_info in units.items():
                # 根据想定文件构造 EntityInfo

                if "rcs" not in unit_info:
                    unit_info["rcs"] = None
                if "heading" not in unit_info:
                    unit_info["heading"] = None
                if "rcs" not in unit_info:
                    unit_info["rcs"] = None
                if "speed" not in unit_info:
                    unit_info["speed"] = None
                if "health" not in unit_info:
                    unit_info["health"] = None
                if "endurance" not in unit_info:
                    unit_info["endurance"] = None
                if "weapons" not in unit_info:
                    unit_info["weapons"] = None
                if "sensors" not in unit_info:
                    unit_info["sensors"] = None

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
                    sensors=unit_info["sensors"]
                )
                entity = self.game_data.add_entity(entity_info, None, color)
                entities.append(entity)
            # Create the player side (e.g., for blue/red teams)
            side = Side(color)
            side.set_entities(entities)
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
                    # position={"x": unit['x'], "y": unit['y'], "z": unit['z']},
                    position=[unit['x'], unit['y'], unit['z']],
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

    def process_commands(self, all_command_list):
        """处理从玩家收到的指令"""
        for player_command_list in all_command_list:
            command_list = player_command_list.get_commands()  # 获取发起指令的实体
            for command in command_list:
                actor = self.game_data.get_entity(command.actor)
                if command.command_type == 'move':
                    # 调用移动系统更新目标位置
                    if actor.get_component(PathfindingComponent) is None:
                        actor.add_component(PathfindingComponent())

                    pathfinding = actor.get_component(PathfindingComponent)
                    movement = actor.get_component(MovementComponent)
                    position = actor.get_component(PositionComponent)

                    if movement and position and pathfinding:
                        # 目标位置更新，触发路径规划
                        if not np.array_equal(movement.target_position, command.target):
                            movement.target_position = np.array(command.target)
                            # 触发路径规划，只在目标发生变化时进行
                            self.pathfinding_system.handle_path_request(
                                actor, movement.target_position)

                # elif command.command_type == 'attack':
                #     # 调用攻击系统执行攻击操作
                #     self.attack_system.process_attack(
                #         command.actor, command.target, command.params['weapon'])

    def update(self, delta_time):
        """更新所有系统的状态"""
        entities = self.game_data.get_all_entities()

        # 更新路径规划和移动系统
        self.movement_system.update(entities, delta_time)

        # 其他系统（如攻击、检测等）更新
        self.attack_system.update(entities)
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
