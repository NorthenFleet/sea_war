import numpy as np
from ..device import *
from ..event_manager import *


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


class Component:
    def __init__(self, **kwargs):
        self.params = kwargs

    def get_param(self, key):
        return self.params.get(key, None)

    def set_param(self, key, value):
        self.params[key] = value



class HealthComponent(Component):
    def __init__(self, max_health):
        super().__init__(max_health=max_health, current_health=max_health)

    def take_damage(self, damage):
        current_health = self.get_param('current_health')
        self.set_param('current_health', max(0, current_health - damage))

    def is_alive(self):
        return self.get_param('current_health') > 0


class PositionComponent(Component):
    def __init__(self, position):
        super().__init__(position=np.array(position))
        # self.x = position[0]
        # self.y = position[1]
        # self.z = position[2]


class MovementComponent(Component):
    def __init__(self, speed, heading):
        super().__init__(speed=speed, heading=np.array(heading), target_position=None)

    def set_target_position(self, target_position):
        self.set_param('target_position', np.array(target_position))


class PathfindingComponent:
    def __init__(self):
        self.global_path = []  # 小地图上的路径节点
        self.local_path = []   # 大地图局部路径的具体点
        self.current_goal = None  # 当前移动目标点


class SensorComponent(Component):
    def __init__(self, sensor_type, detected_targets=None):
        super().__init__(sensor_type=sensor_type, detected_targets=detected_targets)


# 指控系统组件（目标威胁排序和目标分配）
class CommandControlComponent(Component):
    def __init__(self, fire_channels, threaten_priority=None, assigned_targets=None):
        super().__init__(fire_channels=fire_channels,
                         threaten_priority=threaten_priority, assigned_targets=assigned_targets)


class LauncherComponent(Component):
    def __init__(self, weapon_type, ammo_count):
        super().__init__(weapon_type=weapon_type, ammo_count=ammo_count)


class CollisionComponent(Component):
    def __init__(self, entity, target):
        super().__init__(entity=entity, target=target)


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


class AttackComponent():
    def __init__(self, damage, range):
        self.damage = damage
        self.range = range


class DamageOverTimeComponent():
    def __init__(self, damage_per_tick, duration, tick_interval):
        self.damage_per_tick = damage_per_tick  # 每个时间间隔的伤害量
        self.duration = duration  # 持续伤害总时长
        self.tick_interval = tick_interval  # 伤害触发间隔
        self.elapsed_time = 0  # 已经过的时间
        self.time_since_last_tick = 0  # 自上次伤害触发以来过去的时间


class CrashComponent():
    def crash_check(self):
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            for other_id, other_data in self.entities.items():
                if other_id != entity_id:
                    other_position = other_data['position']
                    if np.array_equal(entity_position, other_position):
                        print(f"Entity {entity_id} collided with {other_id}")

        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            if self.map is not None:
                if self.map[int(entity_position[0]), int(entity_position[1])] == 1:
                    print(f"Entity {entity_id} collided with map obstacle")

        map_size = self.map.shape if self.map is not None else None
        for entity_id, entity_data in self.entities.items():
            entity_position = entity_data['position']
            if map_size is not None:
                if (
                    entity_position[0] < 0
                    or entity_position[0] >= map_size[0]
                    or entity_position[1] < 0
                    or entity_position[1] >= map_size[1]
                ):
                    print(f"Entity {entity_id} out of map bounds")
