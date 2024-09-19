from componet import *
from utils import *
import numpy as np


class System:
    def __init__(self):
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)

    def update(self):
        raise NotImplementedError


class MovementSystem(System):
    def update(self, delta_time):
        for entity in self.entities:
            position = entity.get_component(PositionComponent)
            movement = entity.get_component(MovementComponent)
            if position and movement and movement.target_position is not None:
                # 计算向目标位置的向量
                direction = movement.target_position - position.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction /= distance  # 单位向量

                # 更新位置，确保不会超过目标位置
                step_distance = min(distance, movement.speed * delta_time)
                position.position += direction * step_distance

                # 到达目标位置后清除目标
                if np.array_equal(position.position, movement.target_position):
                    movement.target_position = None


class PathfindingSystem(System):
    def update(self):
        for entity in self.entities:
            position = entity.get_component(PositionComponent)
            pathfinding = entity.get_component(PathfindingComponent)
            movement = entity.get_component(MovementComponent)
            if position and pathfinding and movement:
                if pathfinding.current_goal and np.array_equal(position.position, pathfinding.current_goal):
                    if pathfinding.path:
                        pathfinding.current_goal = pathfinding.path.pop(0)
                        movement.set_target(*pathfinding.current_goal)


class AttackSystem(System):
    def update(self, entity, weapon_data):
        for weapon_name in entity.weapon_names:
            # Helper function to find weapon type
            weapon_type = find_weapon_type(weapon_name)
            weapon_params = weapon_data["weapons"][weapon_type][weapon_name]
            # 使用 weapon_params 执行攻击逻辑
            print(
                f"{entity} attacks with {weapon_name}: Damage {weapon_params['damage']}")


class DamageOverTimeSystem(System):
    def update(self, delta_time):
        for entity in self.entities:
            dot = entity.get_component(DamageOverTimeComponent)
            if dot:
                dot.elapsed_time += delta_time
                dot.time_since_last_tick += delta_time

                # 检查是否达到触发伤害的时间间隔
                if dot.time_since_last_tick >= dot.tick_interval:
                    self.apply_damage(entity, dot.damage_per_tick)
                    dot.time_since_last_tick = 0

                # 检查持续伤害是否结束
                if dot.elapsed_time >= dot.duration:
                    entity.remove_component(DamageOverTimeComponent)

    def apply_damage(self, entity, damage):
        health = entity.get_component(HealthComponent)
        if health:
            health.current_health -= damage
            print(
                f"Entity {entity.id} took {damage} damage, remaining health {health.current_health}")
