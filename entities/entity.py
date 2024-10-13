import numpy as np
from device import *


class EntityInfo:
    def __init__(self, side=None, entity_id=None, entity_type=None, position=None, rcs=None, speed=None, heading=None, faction=None, health=None, endurance=None, weapons=None, equipments=None, sensors=None, launcher=None):
        self.side = side
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.position = [position[0], position[1], position[2]]
        self.speed = speed
        self.heading = heading
        self.rcs = rcs
        self.faction = faction
        self.health = health
        self.endurance = endurance
        self.weapons = weapons if weapons is not None else []
        self.sensors = sensors if sensors is not None else []
        self.launcher = launcher


class HealthComponent:
    def __init__(self, hp):
        self.hp = hp


class PositionComponent:
    def __init__(self, position):
        # self.position = np.array([x, y, z])
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]


class MovementComponent:
    def __init__(self, speed, heading):
        self.speed = speed
        self.heading = heading
        self.target_position = None


class PathfindingComponent:
    def __init__(self):
        self.path = []  # 路径上的点
        self.current_goal = None  # 当前移动的目标点


class SensorComponent():
    def __init__(self, name):
        self.name = name


class Entity:
    # 船、飞机、导弹、密集阵、潜艇、
    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.components = {}

    def add_component(self, component):
        self.components[type(component)] = component

    def get_component(self, component_type):
        return self.components.get(component_type)
