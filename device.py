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


class SensorInfo:
    def __init__(self, detect_range) -> None:
        self.detect_range = detect_range


class Sensor:
    def __init__(self, sensor_info):
        self.range = sensor_info.range

    def detect(self, targets):
        detected_targets = detect_targets(self, targets)
        for target in detected_targets:
            print(
                f"Sensor {self.id} detected target {target.id} at ({target.x}, {target.y})")


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
    def __init__(self, damage):
        self.damage = damage

    def explode(self):
        print(f"Ammo explodes with {self.damage} damage")


