import numpy as np
from device import *


class EntityInfo:
    def __init__(self, side=None, entity_id=None, entity_type=None, position=None, rcs=None, speed=None, direction=None, faction=None, hp=None, weapons=None, equipments=None, sensor=None, launcher=None):
        self.side = side
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.position = {
            'x': position["x"], 'y': position["y"], 'z': position["z"]}
        self.speed = speed
        self.direction = direction
        self.rcs = rcs
        self.faction = faction
        self.hp = hp
        self.weapons = weapons if weapons is not None else []
        self.sensor = sensor if sensor is not None else []
        self.launcher = launcher


class Entity:
    def __init__(self, EntityInfo):
        self.id = EntityInfo.entity_id
        self.hp = EntityInfo.hp
        self.type = EntityInfo.entity_type
        self.carrier = None
        self.weapons = {}
        self.sensors = {}
        self.launcher = {}
        self.armo = {}
        self.bullet = []

        self.position = {
            'x': EntityInfo.position["x"], 'y': EntityInfo.position["y"], 'z': EntityInfo.position["z"]}
        self.speed = None
        self.state = None
        self.alive = True

        self.com_level = 5
        self.data_chain = {"Alliance": {}, "Enemy": {}}
        self.detect_entities = {}

    def set_position(self, position):
        self.position = {
            'x': position["x"], 'y': position["y"], 'z': position["z"]}

    def get_position(self):
        return self.position

    def add_weapon(self, weapon):
        self.weapons.append(weapon)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def reset(self, entity_info, device):
        self.entity_info = entity_info
        self.device = device
        self.weapons.clear()
        self.sensors.clear()

    def global_move(self, target_x, target_y, steps):
        self.carrier.global_move(target_x, target_y, steps)

    def local_move(self, angle, speed, steps, time_per_step):
        self.carrier.local_move(angle, speed, steps, time_per_step)

    def detect(self, targets):
        detected_targets = []
        for sensor in self.sensors:
            detected_targets.extend(sensor.detect(targets, self.position))
        return detected_targets

    def attack(self, target, weapon_name=None):
        if weapon_name:
            for weapon in self.weapons:
                if weapon.name == weapon_name:
                    weapon.attack(target)
                    return
            print(f"Weapon {weapon_name} not found.")
        else:
            for weapon in self.weapons:
                weapon.attack(target)

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
