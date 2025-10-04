import json,os
from .core.device import *


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
        self.map_data = []  # 大地图数据
        self.compressed_map = []  # 小地图（压缩后生成）
        self.load_map(path)
        self.compress_map()  # 初始化时生成小地图

    def load_map(self, path):
        """从 json 文件中加载大地图数据"""
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
                self.map_data = data['map_data']
        except FileNotFoundError:
            print(f"Error: File {full_path} not found.")
            return 
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {full_path}.")

    def compress_map(self):
        """压缩生成小地图，每个块用数字表示"""
        small_width = (self.global_width + self.local_block_size - 1) // self.local_block_size
        small_height = (self.global_height + self.local_block_size - 1) // self.local_block_size
        compressed = []

        for y in range(small_height):
            row = []
            for x in range(small_width):
                start_x = x * self.local_block_size
                start_y = y * self.local_block_size
                block = [
                    self.map_data[py][start_x:min(start_x + self.local_block_size, self.global_width)]
                    for py in range(start_y, min(start_y + self.local_block_size, self.global_height))
                ]
                is_obstacle = any(cell == 1 for row in block for cell in row)
                row.append(1 if is_obstacle else 0)
            compressed.append(row)
        self.compressed_map = compressed

    def get_global_position(self, x, y):
        """根据全局坐标计算所属的大格子坐标和局部相对坐标"""
        ix = int(x)
        iy = int(y)
        block_x = ix // self.local_block_size
        block_y = iy // self.local_block_size
        local_x = ix % self.local_block_size
        local_y = iy % self.local_block_size
        return (int(block_x), int(block_y)), (int(local_x), int(local_y))

    def is_global_position_within_bounds(self, x, y):
        """检查全局区域块是否在压缩地图边界内"""
        return 0 <= x < len(self.compress_data[0]) and 0 <= y < len(self.compress_data)

    def is_position_within_bounds(self, x, y):
        """检查给定的全局坐标是否在地图范围内"""
        return 0 <= x < self.global_width and 0 <= y < self.global_height

    def is_obstacle(self, x, y):
        """检查给定全局坐标是否是障碍物"""
        if not self.is_position_within_bounds(x, y):
            return True  # 超出边界视为障碍物
        return self.map_data[y][x] != 0  # 0表示通行

    def get_combined_grid(self, start_block, end_block):
        """获取局部地图区域的矩形组合（起点和终点之间）"""
        min_x = min(start_block[0], end_block[0]) * self.local_block_size
        max_x = (max(start_block[0], end_block[0]) + 1) * self.local_block_size
        min_y = min(start_block[1], end_block[1]) * self.local_block_size
        max_y = (max(start_block[1], end_block[1]) + 1) * self.local_block_size

        combined_grid = [
            row[min_x:max_x] for row in self.map_data[min_y:max_y]
        ]
        return combined_grid

    def display_map(self):
        """打印小地图（压缩地图）"""
        for row in self.compressed_map:
            print(" ".join(map(str, row)))


class Map_back:
    def __init__(self, path):
        self.global_width = 0
        self.global_height = 0
        self.local_block_size = 0
        self.map_data = []  # 存储整体地图数据
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
                self.map_data = data['map_data']  # 整体地图数据
        except FileNotFoundError:
            print(f"Error: File {path} not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {path}.")

    def get_local_grid(self, global_x, global_y):
        """根据全局块坐标获取局部地图"""
        start_x = global_x * self.local_block_size
        start_y = global_y * self.local_block_size
        end_x = start_x + self.local_block_size
        end_y = start_y + self.local_block_size

        if not self.is_global_position_within_bounds(global_x, global_y):
            return None

        # 提取局部地图数据
        return [
            row[start_x:end_x]
            for row in self.map_data[start_y:end_y]
        ]

    def is_global_position_within_bounds(self, x, y):
        """检查全局块坐标是否在地图边界内"""
        return 0 <= x < self.global_width and 0 <= y < self.global_height

    def get_global_position(self, x, y):
        """根据全局坐标计算所属的大格子坐标和局部格子内的坐标"""
        global_x = x // self.local_block_size
        global_y = y // self.local_block_size
        local_x = x % self.local_block_size
        local_y = y % self.local_block_size
        return (global_x, global_y), (local_x, local_y)

    def is_position_within_bounds(self, x, y):
        """检查全局坐标是否在地图范围内"""
        return 0 <= x < self.global_width * self.local_block_size and \
               0 <= y < self.global_height * self.local_block_size

    def is_obstacle(self, x, y):
        """检查全局坐标是否是障碍物"""
        if not self.is_position_within_bounds(x, y):
            return True  # 超出边界视为障碍物

        (global_x, global_y), (local_x, local_y) = self.get_global_position(x, y)
        local_grid = self.get_local_grid(global_x, global_y)

        if local_grid:
            return local_grid[local_y][local_x] != 0  # 假设0表示通行，非0表示障碍物
        return True  # 如果找不到局部块数据，视为障碍物

    def get_combined_grid(self, block, next_block):
        """获取当前块和邻近块组合后的地图"""
        combined_grid = []

        # 当前块和下一个块的范围
        x1, y1 = block
        x2, y2 = next_block

        # 确定组合区域的起点和终点
        start_x = min(x1, x2) * self.local_block_size
        start_y = min(y1, y2) * self.local_block_size
        end_x = (max(x1, x2) + 1) * self.local_block_size
        end_y = (max(y1, y2) + 1) * self.local_block_size

        # 构建组合区域
        for global_y in range(start_y, min(end_y, len(self.map_data))):
            row = []
            for global_x in range(start_x, min(end_x, len(self.map_data[0]))):
                (block_x, block_y), (local_x, local_y) = self.get_global_position(global_x, global_y)
                local_grid = self.get_local_grid(block_x, block_y)
                if local_grid:
                    row.append(local_grid[local_y][local_x])
                else:
                    row.append(1)  # 如果缺失块数据，默认视为障碍物
            combined_grid.append(row)

        return combined_grid

    def display_map(self):
        """打印全局地图的概览"""
        print(f"Map Size: {self.global_width} x {self.global_height} (Blocks)")
        print(f"Each Block Size: {self.local_block_size} x {self.local_block_size}")
        for global_y in range(self.global_height):
            for global_x in range(self.global_width):
                local_grid = self.get_local_grid(global_x, global_y)
                print(f"Block at ({global_x}, {global_y}):")
                for row in local_grid:
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
