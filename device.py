import numpy as np
from utils import *


class Carrier:
    def __init__(self, speed):
        self.speed = speed

    def global_move(self, entity_id, destination):
        direction_vector = np.array(destination) - np.array(self.position)
        distance = np.linalg.norm(direction_vector)
        speed = self.entities[entity_id].get('speed', 1)

        if distance < speed:
            new_position = destination
        else:
            direction_vector_normalized = direction_vector / distance
            new_position = self.position + direction_vector_normalized * speed

        self.entities[entity_id]['position'] = new_position
        print(f"Entity {entity_id} moved to {new_position}")

    def local_move(self, angle, speed, steps, time_per_step):
        path = local_move(self, angle, speed, steps, time_per_step)
        for position in path:
            self.x, self.y = position
            print(f"Carrier {self.id} moved to ({self.x}, {self.y})")


class Sensor:
    def __init__(self, name, detection_range, accurate):
        self.name = name
        self.detection_range = detection_range
        self.accurate = accurate

    def get_range(self):
        return self.detection_range

    def detect(self, targets, position):
        detected_targets = []
        for target in targets:
            distance = ((target.position[0] - position[0]) **
                        2 + (target.position[1] - position[1]) ** 2) ** 0.5
            if distance <= self.range:
                detected_targets.append(target)
        return detected_targets

    def _is_within_range(self, current_position, target_position):
        distance = ((current_position[0] - target_position[0]) ** 2 +
                    (current_position[1] - target_position[1]) ** 2) ** 0.5
        return distance <= self.detection_range


class Launcher:
    def __init__(self, ammo_type, capacity):
        self.ammo_type = ammo_type
        self.capacity = capacity

    def fire(self):
        if self.capacity > 0:
            self.capacity -= 1
            print(
                f"Fired {self.ammo_type}, remaining capacity: {self.capacity}")
        else:
            print("No ammo left to fire")


class Ammo:
    def __init__(self, weapon_type, count):
        self.type = None
        self.count = None
        self.damage = None
        self.range = None

    def set_properties(self, weapon_type, ammo_type, damage, range):
        self.type = weapon_type
        self.ammo_type = ammo_type
        self.damage = damage
        self.range = range

    def fire(self, target):
        if self.count > 0:
            self.count -= 1
            print(
                f"Firing {self.type} at target {target.id}. Remaining: {self.count}")
            return True
        else:
            print(f"No {self.type} left to fire.")
            return False


class Weapon:
    def __init__(self, name, damage, range, cooldown):
        self.name = name
        self.damage = damage
        self.range = range
        self.cooldown = cooldown
        self.current_cooldown = 0

    def attack(self, target):
        if self.current_cooldown == 0:
            target.take_damage(self.damage)
            self.current_cooldown = self.cooldown
        else:
            print(f"{self.name} is cooling down.")

    def reduce_cooldown(self):
        if self.current_cooldown > 0:
            self.current_cooldown -= 1



    
    
