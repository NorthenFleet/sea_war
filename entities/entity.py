import numpy as np
from device import *


class EntityInfo:
    def __init__(self):
        self.side = None
        self.entity_id = None
        self.entity_type = None

        self.position = None
        self.speed = None
        self.direction = None
        self.faction = None
        self.hp = None
        self.attack_power = None
        self.weapons = None
        self.equipments = None
        self.sensor = None
        self.launcher = None


class Entity:
    def __init__(self, EntityInfo):
        self.id = EntityInfo.id
        self.hp = 100
        self.type = None
        self.carrier = Carrier()
        self.sensor = Sensor()
        self.launcher = Launcher()
        self.ammo = Ammo()
        self.state = None
        self.position = np.array(EntityInfo.position)
        self.speed = None
        self.set_observation = []
        self.attack_power = None
        self.alive = True

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
        if not self.alive:
            return
        self.hp -= damage
        if self.hp <= 0:
            self.alive = False
