import numpy as np
from gym import spaces
from env import Env
from game_data import GameData
from init import Map, Device, Side, Scenario
from entities.entity import EntityInfo
from init import Grid, QuadTree
from utils import *
# 定义为游戏的战术层，从战术层面对游戏过程进行解析


class SeaWarEnv(Env):
    def __init__(self, game_config):
        self.name = game_config["name"]
        self.map = None
        self.device_table = None
        self.scenario = None
        self.game_data = GameData()
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

        # 初始化系统
        self.movement_system = MovementSystem(self.game_data)
        self.attack_system = AttackSystem(self.game_data)
        self.detection_system = DetectionSystem(
            self.game_data, self.quad_tree, self.grid)
        self.collision_system = CollisionSystem(self.game_data, self.map)
        self.pathfinding_system = PathfindingSystem(self.game_data)

    def reset_game(self):
        self.current_step = 0
        self.game_over = False
        self.game_data.reset()
        sides = self.load_scenario(self.scenario)
        return self.game_data, sides

    def load_scenario(self, scenario):
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
                # 将实体添加到 game_data 中
                self.game_data.add_entity(entity_info, color)
            # 创建玩家的 Side 对象
            side = Side(color)
            side.set_entities(self.game_data.get_player_unit_ids(color))
            self.sides[color] = side
        return self.sides

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

        self.current_step += 1
