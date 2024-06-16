import numpy as np
from entity import Entity
from entity import Devices
from entity import Equipment


class Ship(Entity):
    def __init__(self, info):
        super().__init__()
        self.carrier(info.speed)
        self.sensor(info.sensor)
        self.launcher(info.launcher)
        self.ammo(info.ammo)
