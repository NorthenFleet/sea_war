import numpy as np
from gym import spaces
from env import Env
from game_data import GameData
from init import Map, Device, Side, Scenario
from entities.entity import EntityInfo
from utils import *
# 定义为游戏的战术层，从战术层面对游戏过程进行解析


class Action():
    def __init__(self, name, unit_id, action_type, target, affliated_id=None):
        self.name = name
        self.unit_id = unit_id
        self.action_type = action_type
        self.affliated_id = affliated_id
        self.target = target
        self.terminated = False


class Action_Manager():
    def __init__(self):
        self.action_list = []

    def add_action(self, action):
        self.action_list.append(action)

    def update(self):
        for action in self.action_list:
            if not action.terminated:
                if action.action_type == "attack":
                    attack(action.unit_id, action.target)
                elif action.action_type == "move":
                    global_move(action.unit_id, action.target)


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

        self.action_manager = Action_Manager()

    def reset_game(self):
        self.current_step = 0
        self.game_over = False
        self.game_data.reset()
        sides = self.load_scenario(self.device_table, self.scenario)
        return self.game_data, sides

    def load_scenario(self, device, scenario):
        for color, unit_list in scenario.data.items():
            for unitid, unit in unit_list.items():
                entity_info = EntityInfo(
                    entity_id=unit['id'],
                    entity_type=unit['entity_type'],
                    position=(unit['x'], unit['y']),
                    speed=unit['speed'],
                    direction=unit['course'],
                    hp=unit['health'],
                    weapons=[w['type'] for w in unit['weapons']],
                    sensor=[s['type'] for s in unit['sensor']]
                )
                self.game_data.add_entity(entity_info, device, color)
            side = Side(color)
            side.set_entities(self.game_data.get_player_unit_ids(color))
            self.sides[color] = side
        return self.sides

    def detect_compute(self):
        visible_entities = {}
        for data in self.game_data.units.items():
            if other_id != entity_id:
                other_position = np.array(data['position'])
                if np.linalg.norm(current_position - other_position) <= detection_range:
                    visible_entities[other_id] = data
        return visible_entities

    def data_chain_compute(self, entity_id):
        pass

    def attack(self, attacker_id, target_id, attack_range):
        if attacker_id not in self.entities or target_id not in self.entities:
            return "Invalid entity"

        attacker = self.entities[attacker_id]
        target = self.entities[target_id]

        attacker_pos = np.array(attacker['position'])
        target_pos = np.array(target['position'])
        if np.linalg.norm(attacker_pos - target_pos) > attack_range:
            return "Target out of range"

        damage = attacker['attack_power']
        target['hp'] -= damage
        if target['hp'] <= 0:
            self.destroy_entity(target_id)
            return f"Target {target_id} destroyed"
        return f"Attacked {target_id}, {damage} damage dealt"

    def crash_check(self):
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            for other_id, other_data in self.entities.items():
                if other_id != entity_id:
                    other_position = other_data['position']
                    if np.array_equal(entity_position, other_position):
                        print(f"Entity {entity_id} collided with {other_id}")

        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            if self.map is not None:
                if self.map[int(entity_position[0]), int(entity_position[1])] == 1:
                    print(f"Entity {entity_id} collided with map obstacle")

        map_size = self.map.shape if self.map is not None else None
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            if map_size is not None:
                if (
                    entity_position[0] < 0
                    or entity_position[0] >= map_size[0]
                    or entity_position[1] < 0
                    or entity_position[1] >= map_size[1]
                ):
                    print(f"Entity {entity_id} out of map bounds")

    def destroy_compute(self):
        pass

    def com_compute(self):
        pass

    def pos_compute(self):
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            entity_speed = entity_data['speed']
            entity_position += entity_speed
            entity_data['position'] = entity_position
            self.entities[entity_id] = entity_data

    def update(self, actions):

        for action in actions:
            if action == 'move':
                # Example move direction
                self.global_move()

            elif action == 'attack':
                self.game_data.units[entity_id]

        # 过程计算
        self.action_manager.update()

        # 状态计算
        self.detect_compute()
        self.com_compute()
        self.destroy_compute()

        self.current_step += 1
