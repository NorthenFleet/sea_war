from event_manager import EventHandler, Event
import numpy as np


class Component:
    pass


class PositionComponent(Component):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MovementComponent(Component):
    def __init__(self, speed):
        self.speed = speed  # 速度，单位可能是每秒像素或其他
        self.target_position = None  # 目标位置

    def set_target(self, x, y):
        self.target_position = np.array([x, y])


def find_path(start, goal, obstacles):
    # 这里使用伪代码来表示，实际中应使用例如A*的算法来找到路径
    path = [start, goal]  # 简化示例，实际应计算考虑障碍物的路径
    return path


class PathfindingComponent(Component):
    def __init__(self):
        self.path = []
        self.current_goal = None

    def update_path(self, start, goal, obstacles):
        self.path = find_path(start, goal, obstacles)
        if self.path:
            self.current_goal = self.path.pop(0)  # 移除起始点


class DetectionComponent(Component):
    def __init__(self, detection_range):
        self.detection_range = detection_range


class WeaponComponent(Component):
    def __init__(self, damage, range):
        self.damage = damage
        self.range = range


class AttackComponent(Component, EventHandler):
    def __init__(self, damage, range, event_manager):
        self.damage = damage
        self.range = range
        self.event_manager = event_manager

    def handle_event(self, event):
        if event.type == 'AttackEvent':
            # 实现被攻击时的逻辑
            self.event_manager.post(
                Event('HealthUpdate', {'damage': event.data['damage']}))


class HealthComponent(Component, EventHandler):
    def __init__(self, max_health, event_manager):
        self.max_health = max_health
        self.current_health = max_health
        self.event_manager = event_manager
        self.event_manager.subscribe('HealthUpdate', self)

    def handle_event(self, event):
        if event.type == 'HealthUpdate':
            self.current_health -= event.data['damage']
            print(f"Updated health: {self.current_health}")


class DamageOverTimeComponent(Component):
    def __init__(self, damage_per_tick, duration, tick_interval):
        self.damage_per_tick = damage_per_tick  # 每个时间间隔的伤害量
        self.duration = duration  # 持续伤害总时长
        self.tick_interval = tick_interval  # 伤害触发间隔
        self.elapsed_time = 0  # 已经过的时间
        self.time_since_last_tick = 0  # 自上次伤害触发以来过去的时间


class CrashComponent(Component):
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
