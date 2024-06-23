import numpy as np
from gym import spaces
from env import Env


class EnvTank(Env):
    def __init__(self, env_config):
        self.name = env_config["name"]
        self.scenario = env_config["scenario"]
        self.map = env_config["map"]
        self.weapon = env_config["weapon"]
        self.players = env_config["players"]
        self.entities = env_config["entities"]
        self.entity_registry = env_config["entity_registry"]
        self.actions = {}
        self.game_over = False
        self.current_step = 0

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

    def reset_game(self):
        self.current_step = 0
        self.game_over = False
        self.entities
        print("Game starts with the following units:")
        return {name: self.observation_space.sample() for name in self.players}

    def create_entity(self, entity_id, entity_type, position, speed, faction, hp, attack_power):
        self.entities[entity_id] = {
            "type": entity_type,
            "position": position,
            "speed": speed,
            "faction": faction,
            "hp": hp,
            "attack_power": attack_power
        }

    def destroy_entity(self, entity_id):
        if entity_id in self.entities:
            del self.entities[entity_id]

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

    def update_hp(self):
        for entity_id, entity_data in self.entities.items():
            entity_hp = entity_data['hp']
            if entity_hp <= 0:
                self.destroy_entity(entity_id)

    def update_posi(self):
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            entity_speed = entity_data['speed']
            entity_position += entity_speed
            entity_data['position'] = entity_position
            self.entities[entity_id] = entity_data

    def update_detect(self):
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            entity_detection_range = entity_data['detection_range']
            visible_entities = self.detect_entities(
                entity_id, entity_detection_range)
            entity_data['visible_entities'] = visible_entities
            self.entities[entity_id] = entity_data

    def update_state(self, actions):
        for entity_id, action in actions.items():
            if action == 'move':
                # Example move direction
                self.global_move()

            elif action == 'attack':
                self.attack(entity_id)

        self.current_step += 1

    def update(self, actions):
        self.update_detect()

        self.update_hp()
        self.update_posi()

        return
