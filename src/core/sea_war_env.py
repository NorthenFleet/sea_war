import numpy as np
from gym import spaces
from .env import Env
from .game_data import GameData
from ..init import Map, DeviceTableDict, Side, Scenario
from .entities.entity import *
from ..init import Grid, QuadTree
from ..utils import *
from .system_manager import *
from .event_manager import EventManager
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

        self.game_map = Map(game_config['map_path'])
        # self.game_map = self.map_process(self.map, (1000, 1000))  # 地图扩充

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

    def map_process(self, original_map, target_size=(1000, 1000)):
        """
        将地图数据从较小的尺寸扩展到目标尺寸 target_size
        :param original_map: 包含地图数据的字典，结构为 {"map_info": ..., "map_data": ...}
        :param target_size: 目标尺寸，默认为(1000, 1000)
        :return: 更新后的 original_map，地图数据已扩展
        """

        # 提取地图矩阵
        map_data = original_map.grid
        original_map_matrix = np.array(map_data)  # 将地图矩阵转换为NumPy数组
        original_height, original_width = original_map_matrix.shape

        # 使用 np.tile 函数来扩展地图
        repeat_factor_height = target_size[0] // original_height
        repeat_factor_width = target_size[1] // original_width

        # 扩展地图
        expanded_map = np.tile(original_map_matrix,
                               (repeat_factor_height, repeat_factor_width))

        # 修剪到目标尺寸（1000x1000），确保边缘对齐
        expanded_map = expanded_map[:target_size[0], :target_size[1]]

        # 更新 original_map 中的 'map_data'
        original_map.grid = expanded_map.tolist()  # 更新地图数据

        return original_map  # 返回更新后的 original_map

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
            detection = SensorComponent(
                entity.entity_type, entity.sensor)
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
                        if not np.array_equal(movement.get_param("target_position"), command.target):
                            movement.set_param("target_position", np.array(
                                command.target))
                            # 触发路径规划，只在目标发生变化时进行
                            self.pathfinding_system.handle_path_request(
                                actor, movement.get_param("target_position"))

                elif command.command_type == 'attack':
                    # 简化的攻击：判距并调用攻击系统或直接施加伤害
                    attacker = actor
                    target = self.game_data.get_entity(command.target)
                    if attacker is None or target is None:
                        continue
                    pos_a = attacker.get_component(PositionComponent)
                    pos_t = target.get_component(PositionComponent)
                    if not pos_a or not pos_t:
                        continue
                    pa = pos_a.get_param('position')[:DD]
                    pt = pos_t.get_param('position')[:DD]
                    dist = np.linalg.norm(pa - pt)
                    launcher = attacker.get_component(LauncherComponent)
                    if launcher:
                        # 目前未集成设备表的射程，先直接允许攻击
                        health = target.get_component(HealthComponent)
                        if health:
                            # 简化：固定伤害或依据设备表后续扩展
                            dmg = 10
                            health.take_damage(dmg)
                elif command.command_type == 'stop':
                    # 停止：清除移动目标并重置路径规划
                    pathfinding = actor.get_component(PathfindingComponent)
                    movement = actor.get_component(MovementComponent)
                    if movement:
                        movement.set_param("target_position", None)
                    if pathfinding:
                        pathfinding.current_goal = None

    def update(self, delta_time):
        """更新所有系统的状态"""
        entities = self.game_data.get_all_entities()

        # 更新路径规划和移动系统
        self.movement_system.update(entities, delta_time)
        self.game_data.distance_table_compute()

        # 更新探测系统
        self.detection_system.update()

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

    def get_serialized_state(self):
        """
        获取可序列化的游戏状态，用于网络传输
        """
        entities_data = []
        for entity in self.game_data.get_all_entities():
            # 获取实体的基本信息
            entity_data = {
                'id': entity.entity_id if hasattr(entity, 'entity_id') else str(id(entity)),
                'type': entity.entity_type if hasattr(entity, 'entity_type') else 'unknown',
                'faction': entity.faction if hasattr(entity, 'faction') else 'neutral'
            }
            
            # 获取位置组件
            position_component = entity.get_component('PositionComponent')
            if position_component:
                pos = position_component.get_position()
                entity_data['position'] = {
                    'x': float(pos[0]) if hasattr(pos, '__getitem__') else 0,
                    'y': float(pos[1]) if hasattr(pos, '__getitem__') and len(pos) > 1 else 0,
                    'z': float(pos[2]) if hasattr(pos, '__getitem__') and len(pos) > 2 else 0
                }
            
            # 获取移动组件
            movement_component = entity.get_component('MovementComponent')
            if movement_component:
                entity_data['speed'] = float(movement_component.get_param('speed', 0))
                direction = movement_component.get_param('direction', [0, 0, 0])
                entity_data['direction'] = {
                    'x': float(direction[0]) if hasattr(direction, '__getitem__') else 0,
                    'y': float(direction[1]) if hasattr(direction, '__getitem__') and len(direction) > 1 else 0,
                    'z': float(direction[2]) if hasattr(direction, '__getitem__') and len(direction) > 2 else 0
                }
            
            # 获取生命值组件
            health_component = entity.get_component('HealthComponent')
            if health_component:
                entity_data['hp'] = float(health_component.get_health())
                entity_data['max_hp'] = float(health_component.get_max_health())
                entity_data['alive'] = health_component.is_alive()
            
            # 获取武器组件
            weapon_component = entity.get_component('WeaponComponent')
            if weapon_component:
                weapons = weapon_component.get_weapons()
                entity_data['weapons'] = [{
                    'id': w.id if hasattr(w, 'id') else str(id(w)),
                    'type': w.type if hasattr(w, 'type') else 'unknown',
                    'ammo': w.ammo if hasattr(w, 'ammo') else 0
                } for w in weapons]
            
            entities_data.append(entity_data)
        
        # 地图数据
        map_data = {
            'width': self.map.width if hasattr(self.map, 'width') else 0,
            'height': self.map.height if hasattr(self.map, 'height') else 0,
            # 可以添加更多地图相关数据
        }
        
        # 游戏状态
        game_state = {
            'entities': entities_data,
            'map': map_data,
            'current_step': self.current_step,
            'game_over': self.game_over
        }
        
        return game_state
