import numpy as np
from device import Carrier, Sensor, Launcher, Ammo


class Devices:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Equipment:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Entity:
    def __init__(self, id):
        self.id = id
        self.carrier = Carrier()
        self.sensor = Sensor()
        self.launcher = Launcher()
        self.ammo = Ammo()
        self.state = None

    def global_move(self, target_x, target_y, steps):
        self.carrier.global_move(target_x, target_y, steps)

    def local_move(self, angle, speed, steps, time_per_step):
        self.carrier.local_move(angle, speed, steps, time_per_step)

    def detect(self, targets):
        self.sensor.detect(targets)

    def fire(self):
        self.launcher.fire()

    def set_observer(self, observer):
        pass

    def change_state(self, state):
        self.state = state

    def take_damage(self, damage):
        pass
