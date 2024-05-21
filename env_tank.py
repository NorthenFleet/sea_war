import numpy as np
from gym import spaces
from env import Env


class EnvTank(Env):
    def __init__(self, game_config):
        self.name = game_config["name"]
        self.scenario = game_config["scenario"]
        self.map = game_config["map"]
        self.weapon = game_config["weapon"]
        self.state = game_config["scenario"]

        self.players = self.scenario.players
        self.entities = {}
        self.game_over = False
        self.current_step = 0

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

    def load_scenario(self, scenario):
        self.scenario = scenario

    def reset_game(self, config):
        self.current_step = 0
        self.game_over = False
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

    def local_move(self, entity_id, move_direction, move_distance=None):
        if entity_id not in self.entities:
            print(f"Entity {entity_id} does not exist.")
            return

        current_position = self.entities[entity_id]['position']
        speed = self.entities[entity_id].get('speed', 1)
        move_distance = move_distance if move_distance is not None else speed

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
        speed = self.entities[entity_id].get('speed', 1)

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

    def update(self, actions):
        for entity_id, action in actions.items():
            if action == 'move':
                # Example move direction
                self.local_move(entity_id, move_direction=(1, 0))
            elif action == 'delete':
                self.destroy_entity(entity_id)

        self.current_step += 1
        if self.current_step > 100:
            self.game_over = True

        return self.detect_entities()
