import numpy as np
from entities.utils import *


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
    def __init__(self, sensor_type):
        config = device_config['sensors'][sensor_type]
        self.type = sensor_type
        self.range = config['range']

    def detect(self, targets, position):
        detected_targets = []
        for target in targets:
            distance = ((target.position[0] - position[0]) **
                        2 + (target.position[1] - position[1]) ** 2) ** 0.5
            if distance <= self.range:
                detected_targets.append(target)
        return detected_targets


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
        config = device_config['weapons'][weapon_type]
        self.type = weapon_type
        self.count = count
        self.damage = config['damage']
        self.range = config['range']

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
    def __init__(self, weapon_type, ammo_type, capacity):
        self.weapon_type = weapon_type
        self.ammo_type = ammo_type
        self.capacity = capacity
