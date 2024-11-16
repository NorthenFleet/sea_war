import json,os
from device import *


class DataLoader:
    def __init__(self, path):
        self.data = self.load_json(path)

    @staticmethod
    def load_json(path):
        # 获取当前脚本的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 组合完整的文件路径
        full_path = os.path.join(script_dir, path)
        try:
            with open(full_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {full_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {full_path}.")
            return None


class Side:
    def __init__(self, name):
        self.name = name
        self.entities = []
        self.enemies = []

    def set_entities(self, entities):
        self.entities = entities

    def add_entity(self, entity):
        self.entities.append(entity)

    def get_entities(self):
        return self.entities

    def set_enemies(self, enemies):
        self.enemies = enemies

    def get_enemies(self):
        return self.enemies


class Scenario(DataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.path = path


class Map:
    def __init__(self, path):
        self.global_width = 0
        self.global_height = 0
        self.local_block_size = 0
        self.map_data = []  # 存储所有的局部地图数据
        self.load_map(path)  # 加载地图数据

    def load_map(self, path):
        """从 json 文件中加载地图数据"""
        # 获取当前脚本的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 组合完整的文件路径
        full_path = os.path.join(script_dir, path)
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
                self.global_width = data['map_info']['global_width']
                self.global_height = data['map_info']['global_height']
                self.local_block_size = data['map_info']['local_block_size']
                self.map_data = data['map_data']  # 加载局部地图数据
        except FileNotFoundError:
            print(f"Error: File {full_path} not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {full_path}.")

    def get_local_grid(self, global_x, global_y):
        """根据全局坐标获取局部地图"""
        for block in self.map_data:
            if block['global_position'] == [global_x, global_y]:
                return block['local_grid']
        return None  # 如果找不到对应的局部块，则返回 None

    def is_position_within_bounds(self, x, y):
        """检查给定的全局坐标是否在地图范围内"""
        return 0 <= x < self.global_width * self.local_block_size and \
            0 <= y < self.global_height * self.local_block_size

    def get_global_position(self, x, y):
        """根据全局坐标计算所属的大格子坐标和小格子内的局部坐标"""
        global_x = x // self.local_block_size
        global_y = y // self.local_block_size
        local_x = x % self.local_block_size
        local_y = y % self.local_block_size
        return (global_x, global_y), (local_x, local_y)

    def is_obstacle(self, x, y):
        """
        检查给定全局坐标是否是障碍物
        """
        if not self.is_position_within_bounds(x, y):
            return True  # 超出边界视为障碍物

        (global_x, global_y), (local_x, local_y) = self.get_global_position(x, y)
        local_grid = self.get_local_grid(global_x, global_y)

        if local_grid is not None:
            return local_grid[local_y][local_x] != 0  # 假设0表示通行，非0表示障碍物
        return True  # 如果没有找到局部块数据，视为障碍物

    def display_map(self):
        """打印全局地图的概览"""
        print(f"Map Size: {self.global_width} x {self.global_height} (Blocks)")
        print(
            f"Each Block Size: {self.local_block_size} x {self.local_block_size}")
        for block in self.map_data:
            global_pos = block["global_position"]
            print(f"Block at {global_pos}:")
            for row in block["local_grid"]:
                print(" ".join(map(str, row)))
            print()  # 空行分隔不同的块


class DeviceTable(DataLoader):
    def __init__(self, path):
        super().__init__(path)

        self.weapons = {}
        self.sensors = {}
        self.launchers = {}

        for weapon_data in self.data['weapons']:
            weapon = Weapon(**weapon_data)
            self.weapons[weapon.name] = weapon
        for sensor_data in self.data['sensors']:
            sensor = Sensor(**sensor_data)
            self.sensors[sensor.name] = sensor

    def get_weapon(self, name):
        return self.weapons.get(name)

    def get_sensor(self, name):
        return self.sensors.get(name)


class DeviceTableDict(DataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.weapons = {}
        self.sensors = {}
        for item in self.data['weapons']:
            weapon_id = item['name']
            if 'guidance_method' not in item:
                item['guidance_method'] = None
            if 'range_min' not in item:
                item['range_min'] = None
            if 'range_max' not in item:
                item['range_max'] = None
            if 'height_min' not in item:
                item['height_min'] = None
            if 'height_max' not in item:
                item['height_max'] = None
            if 'speed' not in item:
                item['speed'] = None
            self.weapons[weapon_id] = {
                'type': item['type'],
                'guidance_method': item['guidance_method'],
                'range_min': item['range_min'],
                'range_max': item['range_max'],
                'height_min': item['height_min'],
                'height_max': item['height_max'],
                'speed': item['speed'],
                'price': item['price'],
                'cooldown': item.get('cooldown', 0)
            }
        for item in self.data['sensors']:
            sensor_id = item['name']
            self.sensors[sensor_id] = {
                'type': item['type'],
                'detection_range': item['detection_range'],
                'height': item.get('height', 0),
                'accurate': item.get('accurate', 0)
            }

    def get_weapon(self, name):
        return self.weapons.get(name)

    def get_sensor(self, name):
        return self.sensors.get(name)


class Grid:
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.cells = [[[] for _ in range(grid_size // cell_size)]
                      for _ in range(grid_size // cell_size)]

    def add_entity(self, entity):
        x, y = int(
            entity.position[0] // self.cell_size), int(entity.position[1] // self.cell_size)
        self.cells[x][y].append(entity)

    def remove_entity(self, entity, old_position):
        x, y = int(
            old_position[0] // self.cell_size), int(old_position[1] // self.cell_size)
        self.cells[x][y].remove(entity)

    def get_nearby_entities(self, entity):
        x, y = int(
            entity.position[0] // self.cell_size), int(entity.position[1] // self.cell_size)
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(self.cells) and 0 <= ny < len(self.cells[0]):
                    nearby.extend(self.cells[nx][ny])
        return nearby


class QuadTree:
    def __init__(self, boundary, capacity):
        # Boundary is a rectangle defined as [x, y, width, height]
        self.boundary = boundary
        self.capacity = capacity
        self.entities = []
        self.divided = False

    def insert(self, entity):
        if not self.in_boundary(entity.position):
            return False
        if len(self.entities) < self.capacity:
            self.entities.append(entity)
            return True
        if not self.divided:
            self.subdivide()
        return (self.northeast.insert(entity) or
                self.northwest.insert(entity) or
                self.southeast.insert(entity) or
                self.southwest.insert(entity))

    def in_boundary(self, position):
        x, y = position
        x0, y0, w, h = self.boundary
        return x0 <= x < x0 + w and y0 <= y < y0 + h

    def subdivide(self):
        x, y, w, h = self.boundary
        hw, hh = w / 2, h / 2
        self.northeast = QuadTree([x + hw, y, hw, hh], self.capacity)
        self.northwest = QuadTree([x, y, hw, hh], self.capacity)
        self.southeast = QuadTree([x + hw, y + hh, hw, hh], self.capacity)
        self.southwest = QuadTree([x, y + hh, hw, hh], self.capacity)
        self.divided = True

        for entity in self.entities:
            self.northeast.insert(entity) or self.northwest.insert(
                entity) or self.southeast.insert(entity) or self.southwest.insert(entity)
        self.entities = []

    def query_circle(self, center, radius):
        """查询在给定圆形区域内的所有实体"""
        found = []
        range_rect = [center[0] - radius, center[1] -
                      radius, 2 * radius, 2 * radius]

        if not self.overlaps(range_rect, self.boundary):
            return found

        for entity in self.entities:
            if np.linalg.norm(entity.position - center) <= radius:
                found.append(entity)

        if self.divided:
            found.extend(self.northeast.query_circle(center, radius))
            found.extend(self.northwest.query_circle(center, radius))
            found.extend(self.southeast.query_circle(center, radius))
            found.extend(self.southwest.query_circle(center, radius))
        return found

    def query_range(self, range_rect):
        """ Return all entities within range_rect, where range_rect is [x, y, width, height] """
        found = []
        if not self.overlaps(range_rect, self.boundary):
            return found
        for entity in self.entities:
            if self.in_range(entity.position, range_rect):
                found.append(entity)
        if self.divided:
            found.extend(self.northeast.query_range(range_rect))
            found.extend(self.northwest.query_range(range_rect))
            found.extend(self.southeast.query_range(range_rect))
            found.extend(self.southwest.query_range(range_rect))
        return found

    def overlaps(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def in_range(self, position, range_rect):
        x, y = position
        rx, ry, rw, rh = range_rect
        return rx <= x < rx + rw and ry <= y < ry + rh


class Radar:
    def __init__(self, radar):
        self.detection_range = radar["detection_range"]
        self.rcs_level = radar["rcs_level"]
