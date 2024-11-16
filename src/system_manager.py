from entities.entity import *
import numpy as np
from event_manager import Event
from utils import *


class System:
    def __init__(self, game_data):
        self.game_data = game_data  # 保存game_data的引用

    def get_all_entities(self):
        """获取游戏数据中的所有实体。"""
        return self.game_data.get_all_entities()


class PositionSystem(System):
    def __init__(self, game_data):
        super().__init__(game_data)

    def get_entity_positions(self):
        """获取所有实体的位置信息"""
        positions = {}
        for entity in self.get_all_entities():
            position = entity.get_component(PositionComponent)
            if position:
                positions[entity.id] = position.position
        return positions

    def set_position(self, entity_id, new_position):
        """设置指定实体的新位置"""
        entity = self.game_data.units.get(entity_id)  # 获取实体
        if entity:
            position = entity.get_component(PositionComponent)
            if position:
                position.position = new_position
                print(f"Entity {entity_id} 位置已更新为 {new_position}")
            else:
                print(f"Entity {entity_id} 不包含 PositionComponent")
        else:
            print(f"实体 {entity_id} 不存在")


class MovementSystem(System):
    def __init__(self, game_data, event_manager):
        super().__init__(game_data)
        self.event_manager = event_manager

    def update(self, entities, delta_time):
        for entity in entities:
            movement = entity.get_component(MovementComponent)
            position = entity.get_component(PositionComponent)
            pathfinding = entity.get_component(PathfindingComponent)

            if movement and position:
                # 如果有路径规划，并且当前路径中有转向点
                if pathfinding is not None and pathfinding.current_goal is not None:

                    # 计算向当前转向点的向量
                    direction = pathfinding.current_goal - \
                        position.get_param("position")[:DD]
                    distance = np.linalg.norm(direction)

                    if distance > 0:
                        # 将 direction 转换为浮点类型，以防止除法引发数据类型冲突
                        direction = direction.astype(np.float64) / distance

                    # 更新位置，确保不会超过转向点
                    step_distance = min(
                        distance, movement.get_param("speed") * delta_time)
                    position.get_param("position")[
                        0] += direction[0] * step_distance
                    position.get_param("position")[
                        1] += direction[1] * step_distance

                    # 检查是否到达转向点
                    if np.linalg.norm(position.get_param("position")[:2] - pathfinding.current_goal) < 0.1:
                        # 如果路径还有剩余转向点，继续移动到下一个转向点
                        if pathfinding.path:
                            pathfinding.current_goal = pathfinding.path.pop(0)
                        else:
                            # 如果路径为空，意味着到达了最终目标
                            movement.target_position = None
                            self.event_manager.post(
                                Event('MoveCompleteEvent', entity, None))


class PathfindingSystem(System):
    def __init__(self, game_data, event_manager, game_map):
        super().__init__(game_data)
        self.event_manager = event_manager
        self.game_map = game_map
        self.last_goal_map = {}  # 记录实体的上次目标，避免重复计算

    def a_star(self, start, goal):
        """A*算法路径规划，结合分层地图结构"""

        # 全局路径规划
        start_global, start_local = self.game_map.get_global_position(
            *start[:2])
        goal_global, goal_local = self.game_map.get_global_position(*goal[:2])

        if start_global != goal_global:
            global_path = self.a_star_global(start_global, goal_global)
            if not global_path:
                return []  # 如果全局路径不可达，返回空路径

            path = []
            for block in global_path:
                # 在每个块中进行局部规划
                local_start = start_local if block == start_global else (0, 0)
                local_goal = goal_local if block == goal_global else (self.game_map.local_block_size - 1,
                                                                      self.game_map.local_block_size - 1)
                local_grid = self.game_map.get_local_grid(*block)
                local_path = self.a_star_local(
                    local_start, local_goal, local_grid)

                if not local_path:
                    return []  # 如果局部路径不可达，返回空路径

                path.extend([(block[0] * self.game_map.local_block_size + lp[0],
                              block[1] * self.game_map.local_block_size + lp[1]) for lp in local_path])
        else:
            # 如果在同一个块内，则直接进行局部规划
            local_grid = self.game_map.get_local_grid(*start_global)
            path = self.a_star_local(start_local, goal_local, local_grid)

        return path

    def a_star_global(self, start_global, goal_global):
        """全局路径规划：只计算跨越的区域块"""
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = set([start_global])
        came_from = {}
        g_score = {start_global: 0}
        f_score = {start_global: heuristic(start_global, goal_global)}

        while open_set:
            current = min(open_set, key=lambda o: f_score.get(o, float('inf')))
            if current == goal_global:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for neighbor in self.get_global_neighbors(current):
                tentative_g_score = g_score[current] + \
                    heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + \
                        heuristic(neighbor, goal_global)
                    open_set.add(neighbor)

        return []

    def a_star_local(self, start, goal, grid):
        """局部路径规划，在每个区域块中寻找路径"""
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = set([start])
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda o: f_score.get(o, float('inf')))
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for neighbor in self.get_local_neighbors(current, grid):
                tentative_g_score = g_score[current] + \
                    heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + \
                        heuristic(neighbor, goal)
                    open_set.add(neighbor)

        return []

    def get_global_neighbors(self, node):
        """获取全局区域块的邻居"""
        neighbors = [
            (node[0], node[1] + 1),
            (node[0] + 1, node[1]),
            (node[0], node[1] - 1),
            (node[0] - 1, node[1])
        ]
        return [n for n in neighbors if self.game_map.is_position_within_bounds(n[0], n[1])]

    def get_local_neighbors(self, node, grid):
        """获取局部地图中节点的邻居"""
        neighbors = [
            (node[0], node[1] + 1),
            (node[0] + 1, node[1]),
            (node[0], node[1] - 1),
            (node[0] - 1, node[1])
        ]
        valid_neighbors = [
            n for n in neighbors if 0 <= n[0] < len(grid[0]) and 0 <= n[1] < len(grid) and grid[n[1]][n[0]] == 0
        ]
        return valid_neighbors

    def handle_path_request(self, entity, target_position):
        """处理路径规划请求"""
        position = entity.get_component(PositionComponent)

        # 如果目标位置没有变化，避免重复规划路径
        if entity.entity_id in self.last_goal_map and np.array_equal(self.last_goal_map[entity.entity_id], target_position):
            return

        # 更新目标位置
        self.last_goal_map[entity.entity_id] = target_position

        if position is not None and target_position is not None:
            # 执行路径规划
            path = self.a_star(position.get_param("position"), target_position)

            if path:
                pathfinding = entity.get_component(PathfindingComponent)
                if pathfinding:
                    pathfinding.path = path
                    pathfinding.current_goal = pathfinding.path.pop(0)


class DetectionSystem(System):
    def __init__(self, game_data, event_manager, device_table, quad_tree, grid):
        super().__init__(game_data)
        self.device_table = device_table
        self.event_manager = event_manager
        self.quad_tree = quad_tree
        self.grid = grid
        r = self.device_table.get_sensor("R4")

    def update(self):
        for entity in self.get_all_entities():
            position = entity.get_component(
                PositionComponent).get_param("position")[:DD]

            sensor = entity.get_component(SensorComponent)
            if sensor:
                sensor_type = sensor.get_param("sensor_type")
                detection = self.device_table.get_sensor(sensor_type)
            else:
                detection = None
            if position is not None and detection is not None:
                # 检测是否有敌人
                for other_entity in self.get_all_entities():
                    if other_entity.entity_id == entity.entity_id:
                        continue
                    other_position = other_entity.get_component(
                        PositionComponent).get_param("position")[:DD]
                    if other_position is not None and np.linalg.norm(other_position - position) <= detection["detection_range"]:
                        # 触发检测事件
                        detection.on_detected(other_entity)


class AttackSystem(System):
    def __init__(self, game_data, event_manager):
        super().__init__(game_data)
        self.event_manager = event_manager

    def update(self, entities):
        for entity in entities:
            weapon = entity.get_component(LauncherComponent)
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
