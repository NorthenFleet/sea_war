import numpy as np


class GameLogic:
    def __init__(self, scenario_config, map_config, weapon_config):
        self.scenario = scenario_config
        self.map = map_config
        self.weapon = weapon_config
        self.entities = {}
        self.current_step = 0
        self.game_over = False

    def load_scenario(self, scenario):
        self.scenario = scenario

    def create_entity(self, entity_id, entity_type, position, speed, faction, hp, attack_power):
        self.entities[entity_id] = {
            "type": entity_type,
            "position": position,
            "speed": speed,
            "faction": faction,
            "hp": hp,
            "attack_power": attack_power
        }

    def delete_entity(self, entity_id):
        if entity_id in self.entities:
            del self.entities[entity_id]

    def local_move(self, entity_id, move_direction, move_distance=None):
        if entity_id not in self.entities:
            print(f"Entity {entity_id} does not exist.")
            return

        current_position = self.entities[entity_id]['position']
        speed = self.entities[entity_id].get('speed', 1)  # 假设实体有速度属性
        move_distance = move_distance if move_distance is not None else speed

        # 计算新位置
        new_position = current_position + \
            np.array(move_direction) * move_distance
        self.entities[entity_id]['position'] = new_position
        print(f"Entity {entity_id} moved locally to {new_position}")

    def global_move(self, entity_id, destination):
        if entity_id not in self.entities:
            print(f"Entity {entity_id} does not exist.")
            return

        current_position = self.entities[entity_id]['position']
        direction_vector = np.array(destination) - np.array(current_position)
        distance = np.linalg.norm(direction_vector)
        speed = self.entities[entity_id].get('speed', 1)  # 假设实体有速度属性

        if distance < speed:
            new_position = destination
        else:
            direction_vector_normalized = direction_vector / distance
            new_position = current_position + direction_vector_normalized * speed

        self.entities[entity_id]['position'] = new_position
        print(f"Entity {entity_id} moved to {new_position}")

    def detect_entities(self, entity_id, detection_range):
        if entity_id not in self.entities:
            return {}

        current_position = np.array(self.entities[entity_id]['position'])
        visible_entities = {}
        for other_id, data in self.entities.items():
            if other_id != entity_id:
                other_position = np.array(data['position'])
                if np.linalg.norm(current_position - other_position) <= detection_range:
                    visible_entities[other_id] = data
        return visible_entities

    def attack(self, attacker_id, target_id, attack_range):
        if attacker_id not in self.entities or target_id not in self.entities:
            return "Invalid entity"

        attacker = self.entities[attacker_id]
        target = self.entities[target_id]

        # 检查目标是否在攻击范围内
        attacker_pos = np.array(attacker['position'])
        target_pos = np.array(target['position'])
        if np.linalg.norm(attacker_pos - target_pos) > attack_range:
            return "Target out of range"

        # 执行攻击
        damage = attacker['attack_power']
        target['hp'] -= damage
        if target['hp'] <= 0:
            self.delete_entity(target_id)
            return f"Target {target_id} destroyed"
        return f"Attacked {target_id}, {damage} damage dealt"

    def step(self, actions):
        # 处理动作，更新状态
        for entity_id, action in actions.items():
            if action == 'move':
                # 假设每次移动改变位置1
                self.move_entity(
                    entity_id, self.entities[entity_id]['position'] + 1)
            elif action == 'delete':
                self.delete_entity(entity_id)

        self.current_step += 1
        if self.current_step > 100:  # 示例结束条件
            self.game_over = True

        return self.detect_entities()
