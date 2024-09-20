import numpy as np


class System:
    def __init__(self, game_data):
        self.game_data = game_data  # 保存game_data的引用

    def get_all_entities(self):
        """获取游戏数据中的所有实体。"""
        return self.game_data.get_all_entities()

    def update(self, delta_time):
        raise NotImplementedError


class MovementSystem(System):
    def __init__(self, game_data):
        super().__init__(game_data)

    def update(self, delta_time):
        for entity in self.get_all_entities():
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

                # 更新 game_data 中的实体状态
                self.game_data.add_entity(entity.id, entity)


class PathfindingSystem(System):
    def __init__(self, game_data):
        super().__init__(game_data)

    def update(self, delta_time):
        for entity in self.get_all_entities():
            position = entity.get_component(PositionComponent)
            pathfinding = entity.get_component(PathfindingComponent)
            movement = entity.get_component(MovementComponent)
            if position and pathfinding and movement:
                if pathfinding.current_goal and np.array_equal(position.position, pathfinding.current_goal):
                    if pathfinding.path:
                        pathfinding.current_goal = pathfinding.path.pop(0)
                        movement.set_target(*pathfinding.current_goal)

                # 更新 game_data 中的实体状态
                self.game_data.add_entity(entity.id, entity)


class DetectionSystem(System):
    def __init__(self, game_data, quad_tree, grid):
        super().__init__(game_data)
        self.quad_tree = quad_tree
        self.grid = grid

    def update(self, delta_time):
        for entity in self.get_all_entities():
            position = entity.get_component(PositionComponent)
            detection = entity.get_component(DetectionComponent)
            if position and detection:
                # 检测是否有敌人
                for other_entity in self.get_all_entities():
                    if other_entity.id == entity.id:
                        continue
                    other_position = other_entity.get_component(
                        PositionComponent)
                    if other_position and np.linalg.norm(other_position.position - position.position) <= detection.radius:
                        # 触发检测事件
                        detection.on_detected(other_entity)


class AttackSystem(System):
    def __init__(self, game_data):
        super().__init__(game_data)

    def update(self, delta_time):
        weapon_data = self.game_data.weapon_data
        for entity in self.get_all_entities():
            for weapon_name in entity.weapon_names:
                weapon_type = find_weapon_type(weapon_name)
                weapon_params = weapon_data["weapons"][weapon_type][weapon_name]
                # 使用 weapon_params 执行攻击逻辑
                print(
                    f"{entity} attacks with {weapon_name}: Damage {weapon_params['damage']}")


class DamageOverTimeSystem(System):
    def __init__(self, game_data):
        super().__init__(game_data)

    def update(self, delta_time):
        for entity in self.get_all_entities():
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

                # 更新 game_data 中的实体状态
                self.game_data.add_entity(entity.id, entity)

    def apply_damage(self, entity, damage):
        health = entity.get_component(HealthComponent)
        if health:
            health.current_health -= damage
            print(
                f"Entity {entity.id} took {damage} damage, remaining health {health.current_health}")


class CollisionSystem(System):
    def __init__(self, game_data, game_map):
        super().__init__(game_data)
        self.game_map = game_map

    def update(self, delta_time):
        for entity in self.get_all_entities():
            position = entity.get_component(PositionComponent)
            collision = entity.get_component(CollisionComponent)
            if position and collision:
                # 检测是否有碰撞
                for other_entity in self.get_all_entities():
                    if other_entity.id == entity.id:
                        continue
                    other_position = other_entity.get_component(
                        PositionComponent)
                    if other_position and np.linalg.norm(other_position.position - position.position) <= collision.radius:
                        # 触发碰撞事件
                        collision.on_collide(other_entity)

                # 检查是否与地图障碍物碰撞
                if self.game_map[int(position.position[0]), int(position.position[1])] == 1:
                    collision.on_collide_map()
