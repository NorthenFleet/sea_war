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

    def move(self, x, y):
        self.carrier.move(x, y)

    def detect(self):
        self.sensor.detect()

    def fire(self):
        self.launcher.fire()

    def update_position(self):
        pass

    def set_observer(self, observer):
        pass

    def update_state(self):
        pass

    def take_damage(self, damage):
        pass
