import numpy as np
import os
import re
import json
from gym import spaces
from .env import Env
from .game_data import GameData
from ..init import Map, DeviceTableDict, Side, Scenario
from .entities.entity import *
from ..init import Grid, QuadTree
from ..utils import *
from .system_manager import *
from .event_manager import EventManager
# 定义为游戏的战术层，从战术层面对游戏过程进行解析


class SeaWarEnv(Env):
    def __init__(self, game_config):
        self.name = game_config["name"]
        self.map = None
        self.scenario = None
        self.default_map_image = None
        self.default_map_json = None

        self.sides = {}
        self.actions = {}
        self.game_over = False
        self.current_step = 0

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        self.game_map = Map(game_config['map_path'])
        self.default_map_json = game_config.get('map_path')
        # self.game_map = self.map_process(self.map, (1000, 1000))  # 地图扩充

        self.device_table = DeviceTableDict(game_config['device_path'])
        self.scenario = Scenario(game_config['scenario_path'])

        self.grid = Grid(1000, 100)
        self.quad_tree = QuadTree([0, 0, 1000, 1000], 4)

        # 游戏逻辑事件系统
        self.event_manager = EventManager()

        self.game_data = GameData(self.event_manager)

        # 初始化系统：移动速度系数可从 game_config 配置，默认 0.5（整体调慢）
        self.movement_speed_factor = float(game_config.get('movement_speed_factor', 0.5))
        # 获取地图边界，默认为1000x1000
        map_bounds = (self.game_map.global_width, self.game_map.global_height) if hasattr(self.game_map, 'global_width') else (1000, 1000)
        self.movement_system = MovementSystem(
            self.game_data, self.event_manager, speed_factor=self.movement_speed_factor, map_bounds=map_bounds)
        self.attack_system = AttackSystem(self.game_data, self.event_manager)
        self.detection_system = DetectionSystem(self.game_data,
                                                self.event_manager, self.device_table, self.quad_tree, self.grid)
        self.pathfinding_system = PathfindingSystem(self.game_data,
                                                    self.event_manager, self.game_map)

        # 组件系统注册到一个列表中，便于统一管理
        self.systems = [
            self.movement_system,
            self.attack_system,
            self.detection_system,
            self.pathfinding_system
        ]

    def map_process(self, original_map, target_size=(1000, 1000)):
        """
        将地图数据从较小的尺寸扩展到目标尺寸 target_size
        :param original_map: 包含地图数据的字典，结构为 {"map_info": ..., "map_data": ...}
        :param target_size: 目标尺寸，默认为(1000, 1000)
        :return: 更新后的 original_map，地图数据已扩展
        """

        # 提取地图矩阵
        map_data = original_map.grid
        original_map_matrix = np.array(map_data)  # 将地图矩阵转换为NumPy数组
        original_height, original_width = original_map_matrix.shape

        # 使用 np.tile 函数来扩展地图
        repeat_factor_height = target_size[0] // original_height
        repeat_factor_width = target_size[1] // original_width

        # 扩展地图
        expanded_map = np.tile(original_map_matrix,
                               (repeat_factor_height, repeat_factor_width))

        # 修剪到目标尺寸（1000x1000），确保边缘对齐
        expanded_map = expanded_map[:target_size[0], :target_size[1]]

        # 更新 original_map 中的 'map_data'
        original_map.grid = expanded_map.tolist()  # 更新地图数据

        return original_map  # 返回更新后的 original_map

    def load_scenario(self, scenario):
        """
        从想定文件中加载场景，并初始化 ECS 实体和组件系统。
        """
        # 绑定场景中的地图信息（如果提供）
        try:
            map_image = scenario.data.get('map_image') or scenario.data.get('map_png') or scenario.data.get('map')
            map_json = scenario.data.get('map_json')
            if isinstance(map_image, str) and map_image.strip():
                self.default_map_image = map_image.strip()
            if isinstance(map_json, str) and map_json.strip():
                self.default_map_json = map_json.strip()
                self.game_map = Map(self.default_map_json)
        except Exception:
            pass

        for color, units in scenario.data.items():
            if color not in ('red', 'blue'):
                continue
            entities = []
            for unit_id, unit_info in units.items():
                # 根据想定文件构造 EntityInfo

                if "rcs" not in unit_info:
                    unit_info["rcs"] = None
                if "heading" not in unit_info:
                    unit_info["heading"] = None
                if "rcs" not in unit_info:
                    unit_info["rcs"] = None
                if "speed" not in unit_info:
                    unit_info["speed"] = None
                if "health" not in unit_info:
                    unit_info["health"] = None
                if "endurance" not in unit_info:
                    unit_info["endurance"] = None
                if "weapons" not in unit_info:
                    unit_info["weapons"] = None
                if "sensors" not in unit_info:
                    unit_info["sensors"] = None

                entity_info = EntityInfo(
                    entity_id=unit_info["id"],
                    side=unit_info["side"],
                    position=[unit_info["x"], unit_info["y"], unit_info["z"]],
                    rcs=unit_info["rcs"],
                    entity_type=unit_info["entity_type"],
                    heading=unit_info["heading"],
                    speed=unit_info["speed"],
                    health=unit_info["health"],
                    endurance=unit_info["endurance"],
                    weapons=unit_info["weapons"],
                    sensors=unit_info["sensors"]
                )
                entity = self.game_data.add_entity(entity_info, None, color)
                entities.append(entity)
            # Create the player side (e.g., for blue/red teams)
            side = Side(color)
            side.set_entities(entities)
            self.sides[color] = side

        return self.game_data

    def load_scenario_OOP(self, scenario):
        """
        Load the scenario into GameData and return the updated GameData object.
        """
        game_data = GameData(
            self.game_data)  # Initialize a new GameData instance

        for color, unit_list in scenario.data.items():
            for unitid, unit in unit_list.items():
                entity_info = EntityInfo(
                    entity_id=unit['id'],
                    entity_type=unit['entity_type'],
                    # position={"x": unit['x'], "y": unit['y'], "z": unit['z']},
                    position=[unit['x'], unit['y'], unit['z']],
                    rcs=unit['rcs'],
                    speed=unit['speed'],
                    direction=unit['course'],
                    hp=unit['health'],
                    weapons=[w['type'] for w in unit['weapons']],
                    sensor=[s['type'] for s in unit['sensor']]
                )
                # Add entity to game_data
                # device=None for now
                game_data.add_entity(entity_info, None, color)

            # Create the player side (e.g., for blue/red teams)
            side = Side(color)
            side.set_entities(game_data.get_player_unit_ids(color))
            self.sides[color] = side

        # Return the populated game_data object
        return game_data

    def game_data_to_component(self, game_data):
        """
        转换 GameData 到 ComponentManager
        """
        # 转换所有实体到 ComponentManager
        for entity_id, entity in game_data.entities.items():
            position = PositionComponent(
                entity_id, entity.position['x'], entity.position['y'], entity.position['z'])
            detection = SensorComponent(
                entity.entity_type, entity.sensor)
            movement = MovementComponent(
                entity_id, entity.speed, entity.direction)
            attack = AttackComponent(entity_id, entity.weapons)
            self.game_data.add_component(entity_id, position)
            self.game_data.add_component(entity_id, detection)
            self.game_data.add_component(entity_id, movement)

    def reset_game(self):
        self.current_step = 0
        self.game_over = False
        self.game_data.reset()
        # 优先从 images 目录生成基于图片的想定
        try:
            generated = self.load_scenario_from_images()
            if generated:
                return self.game_data, self.sides
        except Exception as e:
            print(f"从 images 生成想定失败，回退到 JSON：{e}")

        # 回退：加载 JSON 想定并更新 GameData
        self.game_data = self.load_scenario(self.scenario)
        return self.game_data, self.sides

    def save_state(self, save_path):
        """保存当前游戏状态为JSON。"""
        try:
            state = self.get_serialized_state()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"游戏状态已保存: {save_path}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def load_state(self, save_path):
        """从JSON恢复游戏状态的关键字段（位置、阵营、基础属性）。"""
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # 简单恢复位置与阵营
            id_to_entity = {getattr(ent, 'entity_id', getattr(ent, 'id', id(ent))): ent for ent in self.game_data.get_all_entities()}
            for e in state.get('entities', []):
                eid = e.get('id')
                ent = id_to_entity.get(eid)
                if not ent:
                    continue
                pos_comp = ent.get_component('PositionComponent')
                if pos_comp and 'position' in e:
                    p = e['position']
                    pos_comp.set_position([p.get('x', 0), p.get('y', 0), p.get('z', 0)])
                mov = ent.get_component('MovementComponent')
                if mov and 'direction' in e:
                    d = e['direction']
                    mov.set_param('direction', [d.get('x', 0), d.get('y', 0), d.get('z', 0)])
                if mov and 'speed' in e:
                    mov.set_param('speed', e.get('speed', mov.get_param('speed', 0)))
            print(f"游戏状态已读取: {save_path}")
            return True
        except Exception as e:
            print(f"读取失败: {e}")
            return False

    def load_scenario_from_images(self):
        """扫描 render/images 下的兵力图片，生成初始单位并加入到 GameData。
        支持文件名中携带速度矢量，如：ship_red_10,-5.png 表示 vx=10, vy=-5。
        也支持在 images 目录下提供 units_meta.json 以精确控制单位属性。
        返回 True 表示已生成，False/None 表示无图片或不生成。"""
        # 计算图片目录路径：src/render/images
        render_dir = os.path.join(os.path.dirname(__file__), '..', 'render')
        images_dir = os.path.normpath(os.path.join(render_dir, 'images'))
        if not os.path.isdir(images_dir):
            return False

        # 选择默认地图图片（如存在）
        try:
            map_dir = os.path.normpath(os.path.join(render_dir, 'map'))
            terrain_candidates = [
                'map.png', 'map.jpg', 'map.jpeg', 'map.bmp',
                'terrain.png', 'terrain.jpg', 'terrain.jpeg', 'terrain.bmp',
                'ground.png', 'ground.jpg', 'ground.jpeg', 'ground.bmp',
                '地图.png', '地图.jpg', '地图.jpeg', '地图.bmp'
            ]
            chosen = None
            if os.path.isdir(map_dir):
                for name in terrain_candidates:
                    p = os.path.join(map_dir, name)
                    if os.path.exists(p):
                        chosen = name
                        break
                if chosen is None:
                    for f in os.listdir(map_dir):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                            chosen = f
                            break
            if chosen:
                self.default_map_image = chosen
        except Exception:
            pass

        # 识别的单位类型（需与渲染器的 key 一致）
        known_types = {
            'ship', 'submarine', 'missile', 'ground_based_platforms', 'airport', 'bomber'
        }
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp'}

        # 可选元数据文件
        meta_path = os.path.join(images_dir, 'units_meta.json')
        meta_units = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_units = json.load(f)
            except Exception as e:
                print(f"读取 units_meta.json 失败：{e}")

        # 简单的文件名解析：type_side_vx,vy.*
        def parse_filename(fname):
            name, ext = os.path.splitext(fname)
            parts = name.split('_')
            etype = None
            side = None
            vxvy = None
            for p in parts:
                if p in known_types:
                    etype = p
                elif p.lower() in ('red', 'blue'):
                    side = p.lower()
                else:
                    # 查找形如 12,-5 或 -8,3 的速度串
                    m = re.match(r'^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$', p)
                    if m:
                        vxvy = (float(m.group(1)), float(m.group(2)))
            return etype, side, vxvy

        # 默认速度/朝向配置（当文件名或元数据未指定）
        default_motion = {
            'ship': ((1.0, 0.0), 20.0),
            'submarine': ((0.8, 0.6), 18.0),
            'missile': ((1.0, 1.0), 60.0),
            'ground_based_platforms': ((0.0, 0.0), 0.0),
            'airport': ((0.0, 0.0), 0.0),
            'bomber': ((0.0, 1.0), 30.0)
        }

        # 地图尺寸用于随机/栅格化摆放
        gw = max(1, self.game_map.global_width)
        gh = max(1, self.game_map.global_height)

        created = []
        # 如果有元数据文件，优先按元数据创建
        if isinstance(meta_units, list) and len(meta_units) > 0:
            for u in meta_units:
                etype = u.get('entity_type')
                if etype not in known_types:
                    continue
                side = (u.get('side') or 'blue').lower()
                x = float(u.get('x', gw * 0.5))
                y = float(u.get('y', gh * 0.5))
                z = float(u.get('z', 0.0))
                speed = float(u.get('speed', default_motion[etype][1]))
                heading = u.get('heading')
                if heading is None:
                    vx, vy = default_motion[etype][0]
                    heading = [vx, vy, 0.0]
                # 归一化 heading
                hv = np.array(heading[:2], dtype=np.float64)
                norm = np.linalg.norm(hv)
                if norm > 1e-6:
                    hv = hv / norm
                heading = [float(hv[0]), float(hv[1]), 0.0]
                eid = f"{etype}_{side}_{len(created)+1}"
                entity_info = EntityInfo(
                    entity_id=eid,
                    side=side,
                    position=[x, y, z],
                    rcs=None,
                    entity_type=etype,
                    heading=heading,
                    speed=speed,
                    health=None,
                    endurance=None,
                    weapons=None,
                    sensors=None
                )
                ent = self.game_data.add_entity(entity_info, None, side)
                created.append((side, ent))
        else:
            # 否则根据目录中文件简单生成单位
            files = [f for f in os.listdir(images_dir)
                     if os.path.splitext(f)[1].lower() in image_exts]
            # 将同类型文件分组，便于摆放
            groups = {}
            for f in files:
                etype, side, vxvy = parse_filename(f)
                if etype is None:
                    continue
                groups.setdefault(etype, []).append((f, side, vxvy))

            # 栅格化摆放：不同类型分带，避免重叠
            band_h = gh / (len(groups) + 1) if len(groups) > 0 else gh
            band_i = 1
            for etype, items in groups.items():
                cy = band_h * band_i
                band_i += 1
                cx_step = gw / (len(items) + 1)
                idx = 1
                for fname, side, vxvy in items:
                    cx = cx_step * idx
                    idx += 1
                    z = 0.0
                    # 计算运动参数
                    if vxvy is not None:
                        vx, vy = vxvy
                        speed = float(np.linalg.norm([vx, vy]))
                        if speed < 1e-6:
                            speed = 0.0
                            hv = np.array([0.0, 0.0])
                        else:
                            hv = np.array([vx, vy]) / speed
                    else:
                        hv = np.array(default_motion[etype][0])
                        speed = float(default_motion[etype][1])
                        nrm = np.linalg.norm(hv)
                        if nrm > 1e-6:
                            hv = hv / nrm
                    heading = [float(hv[0]), float(hv[1]), 0.0]

                    # 当文件名未指明阵营，默认为蓝红各生成一例，避免缺失阵营
                    sides_to_create = [side] if side is not None else ['blue', 'red']
                    offset_map = {'blue': 0.0, 'red': cx_step * 0.5}
                    for s in sides_to_create:
                        px = float(cx + (offset_map.get(s, 0.0)))
                        eid = f"{etype}_{s}_{fname}"
                        entity_info = EntityInfo(
                            entity_id=eid,
                            side=s,
                            position=[px, float(cy), z],
                            rcs=None,
                            entity_type=etype,
                            heading=heading,
                            speed=speed,
                            health=None,
                            endurance=None,
                            weapons=None,
                            sensors=None
                        )
                        ent = self.game_data.add_entity(entity_info, None, s)
                        created.append((s, ent))

            # 如果目录下没有任何图片文件，仍生成默认示例单位以便演示
            if len(files) == 0:
                demo_types = [t for t in known_types]
                band_h = gh / (len(demo_types) + 1)
                band_i = 1
                for etype in demo_types:
                    cy = band_h * band_i
                    band_i += 1
                    # 每类放置2个单位，分属不同阵营
                    for idx, side in enumerate(['blue', 'red'], start=1):
                        cx = gw * (idx / 3.0)
                        hv = np.array(default_motion[etype][0])
                        speed = float(default_motion[etype][1])
                        nrm = np.linalg.norm(hv)
                        if nrm > 1e-6:
                            hv = hv / nrm
                        heading = [float(hv[0]), float(hv[1]), 0.0]
                        eid = f"{etype}_{side}_demo{idx}"
                        entity_info = EntityInfo(
                            entity_id=eid,
                            side=side,
                            position=[float(cx), float(cy), 0.0],
                            rcs=None,
                            entity_type=etype,
                            heading=heading,
                            speed=speed,
                            health=None,
                            endurance=None,
                            weapons=None,
                            sensors=None
                        )
                        ent = self.game_data.add_entity(entity_info, None, side)
                        created.append((side, ent))

        if len(created) == 0:
            return False

        # 构建 sides
        by_side = {}
        for side, ent in created:
            by_side.setdefault(side, []).append(ent)
        for color, entities in by_side.items():
            side_obj = Side(color)
            side_obj.set_entities(entities)
            self.sides[color] = side_obj

        # 补全空阵营，避免后续访问 KeyError
        for color in ['red', 'blue']:
            if color not in self.sides:
                self.sides[color] = Side(color)
                self.sides[color].set_entities([])

        return True

    def process_commands(self, all_command_list):
        """处理从玩家收到的指令"""
        for player_command_list in all_command_list:
            command_list = player_command_list.get_commands()  # 获取发起指令的实体
            for command in command_list:
                actor = self.game_data.get_entity(command.actor)
                if command.command_type == 'move':
                    # 调用移动系统更新目标位置
                    if actor.get_component(PathfindingComponent) is None:
                        
                        actor.add_component(PathfindingComponent())

                    pathfinding = actor.get_component(PathfindingComponent)
                    movement = actor.get_component(MovementComponent)
                    position = actor.get_component(PositionComponent)

                    if movement and position and pathfinding:
                        # 目标位置更新，触发路径规划
                        if not np.array_equal(movement.get_param("target_position"), command.target):
                            movement.set_param("target_position", np.array(
                                command.target))
                            # 触发路径规划，只在目标发生变化时进行
                            self.pathfinding_system.handle_path_request(
                                actor, movement.get_param("target_position"))

                elif command.command_type == 'attack':
                    # 简化的攻击：判距并调用攻击系统或直接施加伤害
                    attacker = actor
                    target = self.game_data.get_entity(command.target)
                    if attacker is None or target is None:
                        continue
                    pos_a = attacker.get_component(PositionComponent)
                    pos_t = target.get_component(PositionComponent)
                    if not pos_a or not pos_t:
                        continue
                    pa = pos_a.get_param('position')[:DD]
                    pt = pos_t.get_param('position')[:DD]
                    dist = np.linalg.norm(pa - pt)
                    launcher = attacker.get_component(LauncherComponent)
                    if launcher:
                        # 目前未集成设备表的射程，先直接允许攻击
                        health = target.get_component(HealthComponent)
                        if health:
                            # 简化：固定伤害或依据设备表后续扩展
                            dmg = 10
                            health.take_damage(dmg)
                elif command.command_type == 'set_speed':
                    mv = actor.get_component(MovementComponent)
                    if mv is not None:
                        spd = float(command.params.get('speed', mv.get_param('speed') or 0))
                        # 简单约束，避免过大或负值
                        spd = max(0.0, min(spd, 100.0))
                        mv.set_param('speed', spd)
                elif command.command_type == 'rotate_heading':
                    mv = actor.get_component(MovementComponent)
                    if mv is not None:
                        hv = np.array(mv.get_param('heading') or [1.0, 0.0, 0.0], dtype=np.float64)
                        ang = float(command.params.get('delta_deg', 0.0)) * np.pi / 180.0
                        # 当前朝向角（基于 XY 平面）
                        cur = np.array(hv[:2], dtype=np.float64)
                        nrm = np.linalg.norm(cur)
                        if nrm <= 1e-6:
                            cur = np.array([1.0, 0.0], dtype=np.float64)
                            nrm = 1.0
                        cur = cur / nrm
                        c, s = np.cos(ang), np.sin(ang)
                        rot = np.array([c*cur[0] - s*cur[1], s*cur[0] + c*cur[1]], dtype=np.float64)
                        mv.set_param('heading', np.array([float(rot[0]), float(rot[1]), float(hv[2] if hv.shape[0] > 2 else 0.0)]))
                elif command.command_type == 'sensor_toggle':
                    sensor = actor.get_component(SensorComponent)
                    if sensor is not None:
                        if 'enabled' in sensor.params:
                            # 若携带参数则按参数设置，否则取反
                            if command.params.get('enabled') is None:
                                sensor.set_param('enabled', not bool(sensor.get_param('enabled')))
                            else:
                                sensor.set_param('enabled', bool(command.params.get('enabled')))
                        else:
                            # 默认开启，第一次切换置为 False
                            default_on = True
                            en = command.params.get('enabled', None)
                            sensor.set_param('enabled', default_on if en is None else bool(en))
                elif command.command_type == 'attack_nearest':
                    # 找到距离最近的敌方实体并施加简化伤害
                    # 基于 unit_owner 映射确定敌我
                    owner = self.game_data.unit_owner.get(actor.entity_id, None)
                    pos_a = actor.get_component(PositionComponent)
                    if owner is None or pos_a is None:
                        continue
                    pa = np.array(pos_a.get_param('position')[:DD], dtype=np.float64)
                    nearest = None
                    nearest_dist = 1e9
                    for e in self.game_data.get_all_entities():
                        if e.entity_id == actor.entity_id:
                            continue
                        other_owner = self.game_data.unit_owner.get(e.entity_id, None)
                        if other_owner is None or other_owner == owner:
                            continue
                        hp = e.get_component(HealthComponent)
                        pos_t = e.get_component(PositionComponent)
                        if hp is None or pos_t is None:
                            continue
                        pt = np.array(pos_t.get_param('position')[:DD], dtype=np.float64)
                        d = float(np.linalg.norm(pt - pa))
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest = e
                    if nearest is not None:
                        hp = nearest.get_component(HealthComponent)
                        if hp:
                            hp.take_damage(10)
                elif command.command_type == 'stop':
                    # 停止：清除移动目标并重置路径规划
                    pathfinding = actor.get_component(PathfindingComponent)
                    movement = actor.get_component(MovementComponent)
                    if movement:
                        movement.set_param("target_position", None)
                    if pathfinding:
                        pathfinding.current_goal = None

    def update(self, delta_time):
        """更新所有系统的状态"""
        entities = self.game_data.get_all_entities()

        # 更新路径规划和移动系统
        self.movement_system.update(entities, delta_time)
        self.game_data.distance_table_compute()

        # 更新探测系统
        self.detection_system.update()

        # 其他系统（如攻击、检测等）更新
        self.attack_system.update(entities)
        self.current_step += 1

    def network_update(self):
        if hasattr(self, 'communication_server'):
            # Server-specific logic, e.g., broadcasting collected actions
            pass
        if hasattr(self, 'communication_client'):
            # Client-specific logic, e.g., sending and receiving actions
            if self.communication_client.received_actions:
                # 处理接收到的动作，例如通过事件系统传递给其他系统
                self.event_manager.post(
                    Event('NetworkActionsReceived', self.communication_client.received_actions))

    def get_serialized_state(self):
        """
        获取可序列化的游戏状态，用于网络传输
        """
        entities_data = []
        for entity in self.game_data.get_all_entities():
            # 获取实体的基本信息
            entity_data = {
                'id': entity.entity_id if hasattr(entity, 'entity_id') else str(id(entity)),
                'type': entity.entity_type if hasattr(entity, 'entity_type') else 'unknown',
                'faction': entity.faction if hasattr(entity, 'faction') else 'neutral'
            }
            
            # 获取位置组件
            position_component = entity.get_component('PositionComponent')
            if position_component:
                pos = position_component.get_position()
                entity_data['position'] = {
                    'x': float(pos[0]) if hasattr(pos, '__getitem__') else 0,
                    'y': float(pos[1]) if hasattr(pos, '__getitem__') and len(pos) > 1 else 0,
                    'z': float(pos[2]) if hasattr(pos, '__getitem__') and len(pos) > 2 else 0
                }
            
            # 获取移动组件
            movement_component = entity.get_component('MovementComponent')
            if movement_component:
                entity_data['speed'] = float(movement_component.get_param('speed', 0))
                direction = movement_component.get_param('direction', [0, 0, 0])
                entity_data['direction'] = {
                    'x': float(direction[0]) if hasattr(direction, '__getitem__') else 0,
                    'y': float(direction[1]) if hasattr(direction, '__getitem__') and len(direction) > 1 else 0,
                    'z': float(direction[2]) if hasattr(direction, '__getitem__') and len(direction) > 2 else 0
                }
            
            # 获取生命值组件
            health_component = entity.get_component('HealthComponent')
            if health_component:
                entity_data['hp'] = float(health_component.get_health())
                entity_data['max_hp'] = float(health_component.get_max_health())
                entity_data['alive'] = health_component.is_alive()
            
            # 获取武器组件
            weapon_component = entity.get_component('WeaponComponent')
            if weapon_component:
                weapons = weapon_component.get_weapons()
                entity_data['weapons'] = [{
                    'id': w.id if hasattr(w, 'id') else str(id(w)),
                    'type': w.type if hasattr(w, 'type') else 'unknown',
                    'ammo': w.ammo if hasattr(w, 'ammo') else 0
                } for w in weapons]
            
            entities_data.append(entity_data)
        
        # 地图数据
        map_data = {
            'width': self.map.width if hasattr(self.map, 'width') else 0,
            'height': self.map.height if hasattr(self.map, 'height') else 0,
            # 可以添加更多地图相关数据
        }
        
        # 游戏状态
        game_state = {
            'entities': entities_data,
            'map': map_data,
            'current_step': self.current_step,
            'game_over': self.game_over
        }
        
        return game_state
