from component import *
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
    def __init__(self, game_data, event_manager):
        super().__init__(game_data)
        self.event_manager = event_manager

    def update(self, entities, delta_time):
        for entity in entities:
            movement = entity.get_component(MovementComponent)
            position = entity.get_component(PositionComponent)

            if movement and position and movement.target_position is not None:
                # 计算向目标位置的向量
                direction = movement.target_position - position.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction /= distance  # 单位向量

                # 移动，确保不会超过目标位置
                step_distance = min(distance, movement.speed * delta_time)
                position.position += direction * step_distance

                # 到达目标位置后，清除目标
                if np.array_equal(position.position, movement.target_position):
                    movement.target_position = None

                # 如果完成移动，则触发完成事件
                if movement.target_position is None:
                    self.event_manager.post(
                        Event('MoveCompleteEvent', entity, None))


class PathfindingSystem(System):
    def __init__(self, game_data, event_manager, game_map):
        super().__init__(game_data)
        self.event_manager = event_manager
        self.game_map = game_map  # 用于路径规划的地图数据

    def a_star(self, start, goal):
        # A*算法的简单实现，假设地图是网格，0表示可通行，1表示障碍
        def heuristic(a, b):
            return np.linalg.norm(a - b)

        open_set = set([start])
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda o: f_score.get(o, float('inf')))
            if np.array_equal(current, goal):
                # 还原路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + \
                    heuristic(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + \
                        heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []  # 没有路径找到

    def get_neighbors(self, node):
        # 假设是4方向的网格地图
        neighbors = [
            node + np.array([0, 1]),
            node + np.array([1, 0]),
            node + np.array([0, -1]),
            node + np.array([-1, 0])
        ]
        # 检查邻居是否在地图范围内且没有障碍
        valid_neighbors = [n for n in neighbors if self.is_valid_position(n)]
        return valid_neighbors

    def is_valid_position(self, position):
        # 检查是否是有效位置（没有障碍物）
        x, y = position
        if 0 <= x < self.game_map.width and 0 <= y < self.game_map.height:
            return self.game_map.grid[y][x] == 0  # 0表示无障碍物
        return False

    def update(self, entities):
        for entity in entities:
            pathfinding = entity.get_component(PathfindingComponent)
            position = entity.get_component(PositionComponent)
            movement = entity.get_component(MovementComponent)

            if pathfinding and position and movement:
                if not pathfinding.path and pathfinding.current_goal:
                    # 计算路径
                    pathfinding.path = self.a_star(
                        position.position, pathfinding.current_goal)
                    if pathfinding.path:
                        pathfinding.current_goal = pathfinding.path.pop(0)

                if pathfinding.path:
                    # 继续处理路径中的目标点
                    next_goal = pathfinding.path.pop(0)
                    movement.target_position = next_goal
                else:
                    # 路径完成，触发事件
                    self.event_manager.post(
                        Event('PathCompleteEvent', entity, None))


class DetectionSystem(System):
    def __init__(self, game_data, event_manager, device_table, quad_tree, grid):
        super().__init__(game_data)
        self.device_table = device_table
        self.event_manager = event_manager
        self.quad_tree = quad_tree
        self.grid = grid
        r = self.device_table.get_sensor("R4")

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


class AttackSystem:
    def __init__(self, event_manager):
        self.event_manager = event_manager

    def update(self, entities):
        for entity in entities:
            weapon = entity.get_component(WeaponComponent)
            if weapon:  # 这个实体有武器，可以攻击
                target = self.find_target_in_range(entity)  # 假设有一个找到目标的函数
                if target:
                    # 触发攻击事件
                    event = Event('AttackEvent', source=entity,
                                  target=target, data=weapon.damage)
                    self.event_manager.post(event)

    def find_target_in_range(self, entity, target_list):
        # 找到在射程内的目标（假设实现）
        return target_list  # 返回目标实体


class DamageSystem:
    def __init__(self, event_manager):
        self.event_manager = event_manager
        self.event_manager.register_listener(
            'AttackEvent', self.on_attack_event)

    def on_attack_event(self, event):
        target = event.target
        damage = event.data
        health = target.get_component(HealthComponent)
        if health:
            health.hp -= damage  # 减少目标的生命值
            print(
                f"Entity {target.id} took {damage} damage, remaining HP: {health.hp}")
            if health.hp <= 0:
                print(f"Entity {target.id} is destroyed!")


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
