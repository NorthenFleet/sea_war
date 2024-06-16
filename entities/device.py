from utils import *


class Carrier:
    def __init__(self, id, health, speed):
        self.id = id
        self.health = health
        self.speed = speed

    def global_move(self, target_x, target_y, steps):
        path = global_move(self, target_x, target_y, steps)
        for position in path:
            self.x, self.y = position
            print(f"Carrier {self.id} moved to ({self.x}, {self.y})")

    def local_move(self, angle, speed, steps, time_per_step):
        path = local_move(self, angle, speed, steps, time_per_step)
        for position in path:
            self.x, self.y = position
            print(f"Carrier {self.id} moved to ({self.x}, {self.y})")


class Sensor:
    def __init__(self, range):
        self.range = range

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
