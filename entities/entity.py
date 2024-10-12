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


class HealthComponent:
    def __init__(self, hp):
        self.hp = hp


class PositionComponent:
    def __init__(self, x, y, z):
        self.position = np.array([x, y, z])  # 3D坐标


class MovementComponent:
    def __init__(self, x, y, z, heading):
        self.speed = np.array([x, y, z])  # 3D坐标
        self.heading = heading
        self.target_position = None


class PathfindingComponent:
    def __init__(self):
        self.path = []  # 路径上的点
        self.current_goal = None  # 当前移动的目标点


class SensorComponent():
    def __init__(self, type, range, scope, channel=None):
        self.type = type
        self.range = range
        self.scope = scope
        self.channel = channel


class Entity:
    # 船、飞机、导弹、密集阵、潜艇、
    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.components = {}

    def add_component(self, component):
        self.components[type(component)] = component

    def get_component(self, component_type):
        return self.components.get(component_type, None)
