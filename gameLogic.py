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

    def create_entity(self, entity_id, entity_type, position):
        self.entities[entity_id] = {"type": entity_type, "position": position}

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

    def detect_entities(self):
        # 简化的探测逻辑
        return {eid: ent['position'] for eid, ent in self.entities.items()}

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
