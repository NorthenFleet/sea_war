import pygame
import os, sys
import time
from ..ui.font_loader import load_cn_font
from ..core.entities.entity import *
from ..init import Map
from ..ui.player import CommandList, Command, MoveCommand, AttackCommand, StopCommand, SetSpeedCommand, RotateHeadingCommand, ToggleSensorCommand, AttackNearestCommand


class RenderManager:
    def __init__(self, env, screen_size, tile_size=64, terrain_override=None, show_obstacles=None):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('海战指挥系统')
        self.screen_height = screen_size[1]
        self.screen_width = screen_size[0]
        self.env = env
        self.game_data = env.game_data  # 从环境中获取 game_data
        self.tile_size = tile_size  # 2D 图块的大小
        self.camera_offset = [0, 0]  # 摄像机初始偏移
        self.clock = pygame.time.Clock()
        self.should_close = False
        self.terrain_override = terrain_override  # 指定地图文件名（在 images 目录下）
        
        # 是否显示障碍叠加（绿色网格块）。默认：若无 map_json 则关闭
        if show_obstacles is None:
            self.show_obstacles = bool(getattr(env, 'default_map_json', None))
        else:
            self.show_obstacles = bool(show_obstacles)
        self.sprites = self.load_sprites()  # 加载图片
        
        # 加载地图并计算缩放比例：优先使用环境中的地图对象
        try:
            if getattr(env, 'game_map', None):
                self.map = env.game_map
            else:
                self.map = Map('core/data/map.json')
        except Exception:
            self.map = Map('core/data/map.json')
        
        # 界面布局常量 - 参考军事模拟游戏
        self.TOP_BAR_HEIGHT = 40
        self.BOTTOM_BAR_HEIGHT = 60
        self.RIGHT_PANEL_WIDTH = 280
        self.LEFT_PANEL_WIDTH = 200
        
        # 计算主视图区域
        self.main_view_x = self.LEFT_PANEL_WIDTH
        self.main_view_y = self.TOP_BAR_HEIGHT
        self.main_view_width = self.screen_width - self.LEFT_PANEL_WIDTH - self.RIGHT_PANEL_WIDTH
        self.main_view_height = self.screen_height - self.TOP_BAR_HEIGHT - self.BOTTOM_BAR_HEIGHT
        
        # 计算地图缩放比例（基于主视图区域）
        self.scale_x = self.main_view_width / max(1, self.map.global_width)
        self.scale_y = self.main_view_height / max(1, self.map.global_height)
        
        # 使用中文兼容字体
        self.font = load_cn_font(16)
        self.font_small = load_cn_font(12)
        self.font_large = load_cn_font(20)

        # 选择与交互状态
        self.selected_ids = set()
        self.entity_screen_rects = {}
        self.selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.attack_mode = False

        # 指令与UI动作
        self.command_list = CommandList()
        Command.set_command_list(self.command_list)
        self.ui_actions = []  # e.g., ['pause_toggle', 'speed_up']

        # 界面状态
        self.show_help_overlay = False
        self.current_time = time.time()
        self.mission_time = 0
        
        # 初始化界面组件
        self._init_ui_components()
        
        # 记录所用地图文件名（用于调试显示）
        try:
            self.terrain_filename = os.path.basename(self.sprites.get('terrain_path', ''))
        except Exception:
            self.terrain_filename = None

    def _init_ui_components(self):
        """初始化UI组件布局"""
        # 新的专业军事仿真界面布局
        # 顶部状态栏 - 深蓝色背景，显示时间、状态等信息
        self.top_status_bar = {
            "rect": pygame.Rect(0, 0, self.screen_width, self.TOP_BAR_HEIGHT),
            "bg_color": (25, 45, 85),  # 深蓝色
            "border_color": (60, 80, 120)
        }
        
        # 右侧信息面板 - 单位详情和控制
        self.right_panel = {
            "rect": pygame.Rect(self.screen_width - self.RIGHT_PANEL_WIDTH, self.TOP_BAR_HEIGHT, 
                               self.RIGHT_PANEL_WIDTH, self.screen_height - self.TOP_BAR_HEIGHT - self.BOTTOM_BAR_HEIGHT),
            "bg_color": (20, 30, 50),  # 深蓝灰色
            "border_color": (50, 70, 100)
        }
        
        # 底部控制栏 - 时间轴和播放控制
        self.bottom_control_bar = {
            "rect": pygame.Rect(0, self.screen_height - self.BOTTOM_BAR_HEIGHT, 
                               self.screen_width - self.RIGHT_PANEL_WIDTH, self.BOTTOM_BAR_HEIGHT),
            "bg_color": (25, 45, 85),  # 与顶部一致的深蓝色
            "border_color": (60, 80, 120)
        }
        
        # 左侧信息栏（可选，用于显示事件日志等）
        self.left_info_bar = {
            "rect": pygame.Rect(0, self.TOP_BAR_HEIGHT, self.LEFT_PANEL_WIDTH, 
                               self.screen_height - self.TOP_BAR_HEIGHT - self.BOTTOM_BAR_HEIGHT),
            "bg_color": (20, 30, 50),
            "border_color": (50, 70, 100),
            "visible": False  # 默认隐藏，可通过菜单切换
        }
        
        # 主视图区域（海图显示区）
        left_offset = self.LEFT_PANEL_WIDTH if self.left_info_bar["visible"] else 0
        self.main_view_area = pygame.Rect(
            left_offset, self.TOP_BAR_HEIGHT,
            self.screen_width - self.RIGHT_PANEL_WIDTH - left_offset,
            self.screen_height - self.TOP_BAR_HEIGHT - self.BOTTOM_BAR_HEIGHT
        )
        
        # 小地图位置（在右侧面板顶部）
        minimap_size = 180
        self.minimap_rect = pygame.Rect(
            self.right_panel["rect"].x + 10,
            self.right_panel["rect"].y + 10,
            minimap_size, minimap_size
        )
        
        # 顶部状态栏按钮
        self.top_toolbar_buttons = []
        btn_width, btn_height = 80, 25
        btn_y = 5
        btn_spacing = 85
        
        # 视图控制按钮
        view_buttons = [
            {"label": "视图", "action": "view_menu", "x": 10},
            {"label": "控制", "action": "control_menu", "x": 95},
            {"label": "单位", "action": "units_menu", "x": 180},
            {"label": "帮助", "action": "help_menu", "x": 265}
        ]
        
        for btn in view_buttons:
            rect = pygame.Rect(btn["x"], btn_y, btn_width, btn_height)
            self.top_toolbar_buttons.append({
                "rect": rect,
                "label": btn["label"],
                "action": btn["action"],
                "bg_color": (40, 60, 100),
                "hover_color": (60, 80, 120),
                "text_color": (220, 220, 220)
            })
        
        # 右侧面板控制按钮
        self.right_panel_buttons = []
        btn_start_y = self.minimap_rect.bottom + 20
        btn_width = self.RIGHT_PANEL_WIDTH - 20
        btn_height = 30
        btn_spacing = 35
        
        unit_control_buttons = [
            {"label": "攻击模式", "action": "attack_mode_toggle"},
            {"label": "停止", "action": "issue_stop"},
            {"label": "速度 +", "action": "unit_speed_inc"},
            {"label": "速度 -", "action": "unit_speed_dec"},
            {"label": "左转", "action": "heading_left"},
            {"label": "右转", "action": "heading_right"},
            {"label": "传感器", "action": "sensor_toggle"},
            {"label": "攻击最近", "action": "attack_nearest"}
        ]
        
        for i, btn in enumerate(unit_control_buttons):
            rect = pygame.Rect(
                self.right_panel["rect"].x + 10,
                btn_start_y + i * btn_spacing,
                btn_width, btn_height
            )
            self.right_panel_buttons.append({
                "rect": rect,
                "label": btn["label"],
                "action": btn["action"],
                "bg_color": (30, 50, 80),
                "hover_color": (50, 70, 100),
                "text_color": (200, 200, 200)
            })
        
        # 底部控制栏按钮
        self.bottom_control_buttons = []
        bottom_btn_y = self.bottom_control_bar["rect"].y + 8
        bottom_btn_height = 25
        
        playback_buttons = [
            {"label": "⏸", "action": "pause_toggle", "x": 20, "width": 40},
            {"label": "⏮", "action": "speed_down", "x": 70, "width": 40},
            {"label": "⏭", "action": "speed_up", "x": 120, "width": 40},
            {"label": "重置", "action": "view_reset", "x": 170, "width": 50}
        ]
        
        for btn in playback_buttons:
            rect = pygame.Rect(btn["x"], bottom_btn_y, btn["width"], bottom_btn_height)
            self.bottom_control_buttons.append({
                "rect": rect,
                "label": btn["label"],
                "action": btn["action"],
                "bg_color": (40, 60, 100),
                "hover_color": (60, 80, 120),
                "text_color": (220, 220, 220)
            })
        
        # 合并所有按钮到统一列表（保持兼容性）
        self.buttons = (self.top_toolbar_buttons + 
                       self.right_panel_buttons + 
                       self.bottom_control_buttons)
        
        # 菜单系统保持不变
        self.open_menu = None
        self._menu_items_cache = []
        self.show_help_overlay = False

    def load_sprites(self):
        """加载资源：地形从 render/map 读取，其余单位从 render/images。"""
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        map_dir = os.path.join(os.path.dirname(__file__), 'map')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(map_dir, exist_ok=True)

        # 若用户传入纯色语法（例如：color:#1E90FF 或 color:30,144,255），生成纯色背景
        if isinstance(self.terrain_override, str) and self.terrain_override.lower().startswith('color:'):
            color_spec = self.terrain_override.split(':', 1)[1].strip()
            color = (0, 0, 255)
            try:
                if color_spec.startswith('#') and len(color_spec) in (4, 7):
                    # #RGB 或 #RRGGBB
                    if len(color_spec) == 4:
                        r = int(color_spec[1]*2, 16)
                        g = int(color_spec[2]*2, 16)
                        b = int(color_spec[3]*2, 16)
                    else:
                        r = int(color_spec[1:3], 16)
                        g = int(color_spec[3:5], 16)
                        b = int(color_spec[5:7], 16)
                    color = (r, g, b)
                else:
                    # 逗号分隔的数字，例如 "30,144,255"
                    parts = [int(x.strip()) for x in color_spec.split(',')]
                    if len(parts) == 3:
                        color = tuple(max(0, min(255, v)) for v in parts)
            except Exception:
                pass
            # 创建纯色 Surface
            surface = pygame.Surface((self.screen_width, self.screen_height))
            surface.fill(color)
            sprites = {
                'terrain': surface,
                'terrain_path': f'color:{color[0]},{color[1]},{color[2]}'
            }
            # 其他单位占位按原逻辑继续创建与缩放
            images_dir = os.path.join(os.path.dirname(__file__), 'images')
            os.makedirs(images_dir, exist_ok=True)
            image_paths = {
                'ship': os.path.join(images_dir, 'ship.png'),
                'submarine': os.path.join(images_dir, 'submarine.png'),
                'missile': os.path.join(images_dir, 'missile.png'),
                'ground_based_platforms': os.path.join(images_dir, 'air_defense.png'),
                'airport': os.path.join(images_dir, 'airport.png'),
                'bomber': os.path.join(images_dir, 'bomber.png')
            }
            target_sizes = {
                'ship': (36, 18),
                'submarine': (30, 14),
                'missile': (8, 20),
                'ground_based_platforms': (24, 24),
                'airport': (48, 32),
                'bomber': (40, 40)
            }
            for key, path in image_paths.items():
                if not os.path.exists(path):
                    surface_pl = self._create_placeholder_surface(key)
                    try:
                        pygame.image.save(surface_pl, path)
                    except Exception:
                        pass
            for key, path in image_paths.items():
                try:
                    sprite = pygame.image.load(path).convert_alpha()
                    if key in target_sizes:
                        sprite = pygame.transform.smoothscale(sprite, target_sizes[key])
                    sprites[key] = sprite
                except Exception:
                    sprites[key] = self._create_placeholder_surface(key)
            return sprites

        # 地图图片候选：如果存在则优先作为地形贴图
        terrain_candidates = [
            # 用户指定中文文件名优先
            '六角格地图.jpg',
            # 英文常见命名
            'map.png', 'map.jpg', 'map.jpeg', 'map.bmp',
            'terrain.png', 'terrain.jpg', 'terrain.jpeg', 'terrain.bmp',
            'ground.png',
            # 中文常见命名
            '地图.png', '地图.jpg', '地图.jpeg', '地图.bmp'
        ]
        terrain_path = None
        # 优先使用外部指定
        if self.terrain_override:
            override_path = os.path.join(map_dir, self.terrain_override)
            if os.path.exists(override_path):
                terrain_path = override_path
            else:
                # 递归搜索子目录（允许仅提供文件名）
                for root, _, files in os.walk(map_dir):
                    for f in files:
                        if f == self.terrain_override or os.path.join(root, f).endswith(self.terrain_override):
                            terrain_path = os.path.join(root, f)
                            break
                    if terrain_path:
                        break
        # 其次使用候选列表（当未选择或不存在时）
        if terrain_path is None:
            for name in terrain_candidates:
                p = os.path.join(map_dir, name)
                if os.path.exists(p):
                    terrain_path = p
                    break
                # 子目录递归搜索
                for root, _, files in os.walk(map_dir):
                    if name in files:
                        terrain_path = os.path.join(root, name)
                        break
                if terrain_path:
                    break
        if terrain_path is None:
            terrain_path = os.path.join(map_dir, 'ground.png')

        image_paths = {
            'terrain': terrain_path,
            'ship': os.path.join(images_dir, 'ship.png'),
            'submarine': os.path.join(images_dir, 'submarine.png'),
            'missile': os.path.join(images_dir, 'missile.png'),
            'ground_based_platforms': os.path.join(images_dir, 'air_defense.png'),
            'airport': os.path.join(images_dir, 'airport.png'),
            'bomber': os.path.join(images_dir, 'bomber.png')
        }

        target_sizes = {
            'ship': (36, 18),
            'submarine': (30, 14),
            'missile': (8, 20),
            'ground_based_platforms': (24, 24),
            'airport': (48, 32),
            'bomber': (40, 40)
        }

        sprites = {}

        # 生成并保存占位资源（若缺失）
        for key, path in image_paths.items():
            if not os.path.exists(path):
                surface = self._create_placeholder_surface(key)
                try:
                    pygame.image.save(surface, path)
                except Exception:
                    pass

        # 加载资源
        for key, path in image_paths.items():
            try:
                sprite = pygame.image.load(path).convert_alpha()
                # 地形贴图拉伸到窗口大小
                if key == 'terrain':
                    sprite = pygame.transform.smoothscale(sprite, (self.screen_width, self.screen_height))
                # 其他单位按目标尺寸缩放
                elif key in target_sizes:
                    sprite = pygame.transform.smoothscale(sprite, target_sizes[key])
                sprites[key] = sprite
            except Exception:
                # 兜底占位
                sprites[key] = self._create_placeholder_surface(key)
        # 将地形实际路径保存，供调试显示用
        sprites['terrain_path'] = terrain_path

        return sprites

    def _create_placeholder_surface(self, key):
        # 为不同类型创建更醒目的占位图
        if key == 'terrain':
            w, h = self.screen.get_size()
            surface = pygame.Surface((w, h))
            surface.fill((34, 139, 34))
            return surface

        size_map = {
            'ship': (36, 18),
            'submarine': (30, 14),
            'missile': (8, 20),
            'ground_based_platforms': (24, 24),
            'airport': (48, 32),
            'bomber': (40, 40)
        }
        color_map = {
            'ship': (70, 130, 180),
            'submarine': (72, 61, 139),
            'missile': (220, 20, 60),
            'ground_based_platforms': (255, 215, 0),
            'airport': (105, 105, 105),
            'bomber': (255, 140, 0)
        }
        w, h = size_map.get(key, (28, 28))
        color = color_map.get(key, (200, 200, 200))
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        surface.fill(color)
        # 简单形状增强辨识度
        if key in ('ship', 'submarine'):
            pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 1)
        if key == 'missile':
            pygame.draw.polygon(surface, (255, 255, 255), [(4, 0), (8, 10), (0, 10)])
        return surface

    def isometric_transform(self, x, y):
        """将 2D 坐标转换为等轴测 2.5D 坐标"""
        iso_x = (x - y) * (self.tile_size // 2)
        iso_y = (x + y) * (self.tile_size // 4)
        return iso_x, iso_y

    def draw_terrain(self):
        """绘制地形：显示地图图片，仅对障碍块做半透明叠加，避免整屏绿色覆盖"""
        # 背景地图
        self.screen.blit(self.sprites['terrain'], (0, 0))
        # 如果不显示障碍，直接返回
        if not self.show_obstacles:
            return
        # 绘制障碍半透明叠加
        compressed = self.map.compressed_map
        if not compressed:
            return
        small_height = len(compressed)
        small_width = len(compressed[0]) if small_height > 0 else 0
        if small_width == 0:
            return

        block_pixel_w = max(1, int(self.map.local_block_size * self.scale_x))
        block_pixel_h = max(1, int(self.map.local_block_size * self.scale_y))

        overlay = pygame.Surface((small_width * block_pixel_w, small_height * block_pixel_h), pygame.SRCALPHA)
        obstacle_color_rgba = (0, 100, 0, 120)  # 绿色半透明表示障碍
        for by in range(small_height):
            for bx in range(small_width):
                if compressed[by][bx] == 1:
                    px = int(bx * block_pixel_w)
                    py = int(by * block_pixel_h)
                    pygame.draw.rect(overlay, obstacle_color_rgba, (px, py, block_pixel_w, block_pixel_h))
        # 叠加到屏幕（考虑摄像机偏移）
        self.screen.blit(overlay, (self.camera_offset[0], self.camera_offset[1]))

        # 网格线（可选，弱化显示）
        grid_color = (0, 80, 0)
        for by in range(small_height + 1):
            y = int(by * block_pixel_h) + self.camera_offset[1]
            pygame.draw.line(self.screen, grid_color, (self.camera_offset[0], y), (self.camera_offset[0] + small_width * block_pixel_w, y), 1)
        for bx in range(small_width + 1):
            x = int(bx * block_pixel_w) + self.camera_offset[0]
            pygame.draw.line(self.screen, grid_color, (x, self.camera_offset[1]), (x, self.camera_offset[1] + small_height * block_pixel_h), 1)

    def metric_transform(self, x, y):
        """将地图坐标缩放到屏幕像素坐标（左上为原点）"""
        sx = int(x * self.scale_x)
        sy = int(y * self.scale_y)
        return sx, sy

    def draw_units(self):
        """绘制单位图层"""
        self.entity_screen_rects = {}
        for entity in self.game_data.get_all_entities():
            position = entity.get_component(PositionComponent)
            if position:
                pos = position.get_param('position')
                x, y = pos[0], pos[1]
                px, py = self.metric_transform(x, y)
                sprite = self.sprites.get(entity.entity_type)
                if sprite is None:
                    # 未知类型占位
                    sprite = pygame.Surface((28, 28), pygame.SRCALPHA)
                    sprite.fill((200, 200, 200))
                rect = sprite.get_rect()
                rect.topleft = (px + self.camera_offset[0], py + self.camera_offset[1])
                self.screen.blit(sprite, rect.topleft)
                self.entity_screen_rects[entity.entity_id] = rect.copy()
                # 标签显示，便于确认元素存在
                label = self.font.render(str(entity.entity_type), True, (255, 255, 255))
                self.screen.blit(label, (rect.left, rect.top - 12))
                # 选中高亮
                if entity.entity_id in self.selected_ids:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect, 2)

                # 绘制血量条
                hp_comp = entity.get_component(HealthComponent)
                if hp_comp:
                    max_hp = hp_comp.get_param('max_health')
                    cur_hp = hp_comp.get_param('current_health')
                    if max_hp and max_hp > 0:
                        ratio = max(0.0, min(1.0, cur_hp / max_hp))
                        bar_w = rect.width
                        bar_h = 4
                        bar_x = rect.left
                        bar_y = rect.top - 6
                        pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
                        fg_w = int(bar_w * ratio)
                        color = (40, 200, 40) if ratio > 0.5 else (200, 180, 40) if ratio > 0.2 else (200, 40, 40)
                        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fg_w, bar_h))

        # 框选矩形可视化
        if self.selection_rect:
            pygame.draw.rect(self.screen, (0, 200, 0), self.selection_rect, 1)

    def draw_buttons(self):
        for b in self.buttons:
            pygame.draw.rect(self.screen, (30, 30, 30), b["rect"])  # 背景
            pygame.draw.rect(self.screen, (180, 180, 180), b["rect"], 1)
            label = self.font.render(b["label"], True, (255, 255, 255))
            self.screen.blit(label, (b["rect"].x + 6, b["rect"].y + 4))

    def draw_minimap(self):
        """绘制小地图（位于右上角）"""
        # 小地图位置和尺寸（右上角独立位置）
        mini_w, mini_h = 180, 135
        mini_x = self.screen_width - mini_w - 15  # 距离右边缘15像素
        mini_y = self.TOP_BAR_HEIGHT + 10  # 在顶部状态栏下方10像素
        mini_rect = pygame.Rect(mini_x, mini_y, mini_w, mini_h)
        
        # 小地图背景（深色军事风格）
        pygame.draw.rect(self.screen, (8, 15, 25), mini_rect)
        pygame.draw.rect(self.screen, (40, 60, 80), mini_rect, 2)
        
        # 小地图标题
        title = self.font.render("战术地图", True, (200, 200, 200))
        self.screen.blit(title, (mini_x, mini_y - 25))

        # 绘制地形简化（障碍/地面）
        if self.show_obstacles:
            compressed = self.map.compressed_map
            if compressed:
                small_h = len(compressed)
                small_w = len(compressed[0]) if small_h > 0 else 0
                if small_w > 0:
                    # 将小地图像素映射到迷你地图区域
                    cell_w = mini_w / small_w
                    cell_h = mini_h / small_h
                    for by in range(small_h):
                        for bx in range(small_w):
                            # 使用更符合海战主题的颜色
                            color = (25, 40, 15) if compressed[by][bx] == 1 else (15, 35, 60)  # 陆地/海洋
                            px = int(mini_rect.x + bx * cell_w)
                            py = int(mini_rect.y + by * cell_h)
                            pygame.draw.rect(self.screen, color, (px, py, int(cell_w)+1, int(cell_h)+1))

        # 绘制单位点（更明显的标记）
        mini_scale_x = mini_w / max(1, self.map.global_width)
        mini_scale_y = mini_h / max(1, self.map.global_height)
        for entity in self.game_data.get_all_entities():
            pos_comp = entity.get_component(PositionComponent)
            if not pos_comp:
                continue
            pos = pos_comp.get_param('position')
            ex = int(mini_rect.x + pos[0] * mini_scale_x)
            ey = int(mini_rect.y + pos[1] * mini_scale_y)
            owner = self.game_data.get_unit_owner(entity.entity_id)
            
            # 使用更明显的军事标记
            if owner == 'red':
                color = (255, 60, 60)
                # 绘制敌方单位（方形）
                pygame.draw.rect(self.screen, color, (ex-2, ey-2, 4, 4))
            else:
                color = (60, 150, 255)
                # 绘制友方单位（圆形）
                pygame.draw.circle(self.screen, color, (ex, ey), 3)
            
            # 如果是选中单位，添加高亮边框
            if hasattr(self, 'selected_entities') and entity.entity_id in self.selected_entities:
                pygame.draw.circle(self.screen, (255, 255, 0), (ex, ey), 5, 1)

        # 当前视窗在迷你地图上的矩形（更明显的视窗指示器）
        view_w = int(self.screen_width / max(1, self.scale_x) * mini_scale_x)
        view_h = int(self.screen_height / max(1, self.scale_y) * mini_scale_y)
        view_x = int(mini_rect.x - (self.camera_offset[0] / max(1, self.scale_x)) * mini_scale_x)
        view_y = int(mini_rect.y - (self.camera_offset[1] / max(1, self.scale_y)) * mini_scale_y)
        
        # 限制视窗矩形在小地图范围内
        view_rect = pygame.Rect(view_x, view_y, view_w, view_h)
        clipped_rect = view_rect.clip(mini_rect)
        if clipped_rect.width > 0 and clipped_rect.height > 0:
            pygame.draw.rect(self.screen, (255, 255, 100), clipped_rect, 2)
        
        # 存储小地图矩形供点击检测使用
        self.minimap_rect = mini_rect

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_close = True
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    self.show_obstacles = not self.show_obstacles
                elif event.key == pygame.K_F1:
                    self.show_help_overlay = not self.show_help_overlay
                elif event.key == pygame.K_LEFT:
                    self.camera_offset[0] += 20
                elif event.key == pygame.K_RIGHT:
                    self.camera_offset[0] -= 20
                elif event.key == pygame.K_UP:
                    self.camera_offset[1] += 20
                elif event.key == pygame.K_DOWN:
                    self.camera_offset[1] -= 20

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self._handle_left_click(event.pos)
                elif event.button == 3:  # 右键
                    self._handle_right_click(event.pos)

    def _handle_left_click(self, pos):
        """处理左键点击"""
        # 1. 检查顶部工具栏按钮
        for btn in self.top_toolbar_buttons:
            if btn["rect"].collidepoint(pos):
                action = btn["action"]
                if action.endswith("_menu"):
                    # 切换菜单显示
                    if self.open_menu == action:
                        self.open_menu = None
                    else:
                        self.open_menu = action
                else:
                    self.process_ui_action(action)
                return

        # 2. 检查下拉菜单项
        if self.open_menu and hasattr(self, '_menu_items_cache'):
            for item in self._menu_items_cache:
                if "rect" in item and item["rect"].collidepoint(pos):
                    self.process_ui_action(item["action"])
                    self.open_menu = None  # 关闭菜单
                    return

        # 3. 检查右侧面板按钮
        for btn in self.right_panel_buttons:
            if btn["rect"].collidepoint(pos):
                self.process_ui_action(btn["action"])
                return

        # 4. 检查底部控制栏按钮
        for btn in self.bottom_control_buttons:
            if btn["rect"].collidepoint(pos):
                self.process_ui_action(btn["action"])
                return

        # 5. 检查小地图点击
        if self.minimap_rect.collidepoint(pos):
            self._handle_minimap_click(pos)
            return

        # 6. 主视图区域的单位选择
        if self.main_view_area.collidepoint(pos):
            self._handle_main_view_click(pos)
        
        # 7. 点击其他区域关闭菜单
        self.open_menu = None

    def _handle_right_click(self, pos):
        """处理右键点击"""
        # 只在主视图区域处理右键
        if not self.main_view_area.collidepoint(pos):
            return
            
        # 转换为世界坐标
        world_x = (pos[0] - self.main_view_area.x - self.camera_offset[0]) / self.scale_x
        world_y = (pos[1] - self.main_view_area.y - self.camera_offset[1]) / self.scale_y
        
        if self.attack_mode:
            # 攻击模式：对选中单位下达攻击指令
            target_entity = self._get_entity_at_position(pos)
            if target_entity:
                for entity in self.game_data.get_all_entities():
                    if entity.entity_id in self.selected_ids:
                        AttackCommand(entity.entity_id, target_entity.entity_id)
        else:
            # 移动模式：对选中单位下达移动指令
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    MoveCommand(entity.entity_id, (world_x, world_y))

    def _handle_minimap_click(self, pos):
        """处理小地图点击"""
        # 计算在小地图中的相对位置
        rel_x = pos[0] - self.minimap_rect.x
        rel_y = pos[1] - self.minimap_rect.y
        
        # 转换为世界坐标比例
        map_ratio_x = rel_x / self.minimap_rect.width
        map_ratio_y = rel_y / self.minimap_rect.height
        
        # 更新摄像机位置
        world_width = self.map.width if hasattr(self.map, 'width') else 1000
        world_height = self.map.height if hasattr(self.map, 'height') else 1000
        
        target_world_x = map_ratio_x * world_width
        target_world_y = map_ratio_y * world_height
        
        # 居中显示
        self.camera_offset[0] = -(target_world_x * self.scale_x - self.main_view_area.width // 2)
        self.camera_offset[1] = -(target_world_y * self.scale_y - self.main_view_area.height // 2)

    def _handle_main_view_click(self, pos):
        """处理主视图区域点击"""
        # 转换为相对于主视图的坐标
        view_x = pos[0] - self.main_view_area.x
        view_y = pos[1] - self.main_view_area.y
        
        # 检查是否点击了单位
        clicked_entity = None
        for entity_id, rect in self.entity_screen_rects.items():
            # 调整rect坐标到主视图坐标系
            adjusted_rect = pygame.Rect(
                rect.x - self.main_view_area.x,
                rect.y - self.main_view_area.y,
                rect.width, rect.height
            )
            if adjusted_rect.collidepoint(view_x, view_y):
                clicked_entity = entity_id
                break

        # 处理选择逻辑
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            # Ctrl+点击：切换选择
            if clicked_entity:
                if clicked_entity in self.selected_ids:
                    self.selected_ids.remove(clicked_entity)
                else:
                    self.selected_ids.add(clicked_entity)
        else:
            # 普通点击：替换选择
            self.selected_ids.clear()
            if clicked_entity:
                self.selected_ids.add(clicked_entity)

    def _get_entity_at_position(self, pos):
        """获取指定位置的实体"""
        for entity_id, rect in self.entity_screen_rects.items():
            if rect.collidepoint(pos):
                for entity in self.game_data.get_all_entities():
                    if entity.entity_id == entity_id:
                        return entity
        return None

    def consume_commands(self):
        """返回并清空当前收集的指令列表。"""
        cl = self.command_list
        self.command_list = CommandList()
        Command.set_command_list(self.command_list)
        return cl

    def consume_ui_actions(self):
        actions = self.ui_actions[:]
        self.ui_actions.clear()
        return actions

    def update(self):
        """主渲染循环"""
        # 0. 处理事件
        self.handle_events()
        if self.should_close:
            # 结束渲染循环，让外层游戏逻辑退出
            pygame.display.quit()
            return False
            
        # 1. 清屏
        self.screen.fill((15, 25, 45))  # 深蓝色背景，符合军事仿真风格
        
        # 2. 绘制主视图区域（海图）
        # 设置裁剪区域为主视图
        main_view_clip = self.main_view_area
        self.screen.set_clip(main_view_clip)
        
        # 绘制地形和单位（在主视图区域内）
        self.draw_terrain()
        self.draw_units()
        
        # 取消裁剪
        self.screen.set_clip(None)
        
        # 3. 绘制UI界面组件
        self.draw_top_status_bar()
        self.draw_right_panel()
        self.draw_bottom_control_bar()
        
        # 4. 绘制左侧面板（如果可见）
        if self.left_info_bar["visible"]:
            self.draw_left_info_panel()
        
        # 5. 绘制按钮
        self.draw_modern_buttons()
        
        # 6. 绘制小地图
        self.draw_minimap()
        
        # 7. 绘制菜单（如果打开）
        self.draw_dropdown_menus()
        
        # 8. 显示帧率和调试信息
        if hasattr(self, 'show_debug') and self.show_debug:
            fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (255, 255, 255))
            self.screen.blit(fps_text, (self.screen_width - fps_text.get_width() - 12, self.screen_height - fps_text.get_height() - 12))

        # 9. 刷新显示
        pygame.display.flip()

        # 控制帧率
        self.clock.tick(60)
        return True

    def draw_left_info_panel(self):
        """绘制左侧信息面板"""
        panel = self.left_info_bar
        # 背景
        pygame.draw.rect(self.screen, panel["bg_color"], panel["rect"])
        pygame.draw.rect(self.screen, panel["border_color"], panel["rect"], 1)
        
        # 面板标题
        title = self.font.render("事件日志", True, (220, 220, 220))
        self.screen.blit(title, (panel["rect"].x + 10, panel["rect"].y + 10))
        
        # 模拟事件日志内容
        log_entries = [
            "系统启动完成",
            "地图加载成功", 
            "单位初始化完成",
            "等待指令..."
        ]
        
        y_offset = panel["rect"].y + 40
        for entry in log_entries:
            text = self.font.render(entry, True, (180, 180, 180))
            self.screen.blit(text, (panel["rect"].x + 15, y_offset))
            y_offset += 20

    def draw_top_status_bar(self):
        """绘制顶部状态栏"""
        bar = self.top_status_bar
        # 背景
        pygame.draw.rect(self.screen, bar["bg_color"], bar["rect"])
        pygame.draw.rect(self.screen, bar["border_color"], bar["rect"], 1)
        
        # 标题文字
        title_text = self.font.render("海战指挥系统", True, (220, 220, 220))
        self.screen.blit(title_text, (bar["rect"].x + 350, bar["rect"].y + 8))
        
        # 状态信息
        status_items = [
            f"时间: {time.strftime('%H:%M:%S')}",
            f"选中: {len(self.selected_ids)}个单位",
            f"攻击模式: {'开启' if self.attack_mode else '关闭'}"
        ]
        
        x_offset = self.screen_width - 400
        for i, item in enumerate(status_items):
            text = self.font.render(item, True, (200, 200, 200))
            self.screen.blit(text, (x_offset, bar["rect"].y + 8))
            x_offset += 120

    def draw_right_panel(self):
        """绘制右侧信息面板"""
        panel = self.right_panel
        # 背景
        pygame.draw.rect(self.screen, panel["bg_color"], panel["rect"])
        pygame.draw.rect(self.screen, panel["border_color"], panel["rect"], 1)
        
        # 面板标题
        title = self.font.render("单位控制", True, (220, 220, 220))
        self.screen.blit(title, (panel["rect"].x + 10, panel["rect"].y + 200))
        
        # 选中单位详细信息
        info_y = panel["rect"].y + 225
        if self.selected_ids:
            count = 0
            for entity in self.game_data.get_all_entities():
                if entity.entity_id not in self.selected_ids or count >= 8:
                    continue
                    
                # 单位基本信息
                pos = entity.get_component(PositionComponent)
                mv = entity.get_component(MovementComponent)
                hp = entity.get_component(HealthComponent)
                owner = self.game_data.get_unit_owner(entity.entity_id)
                
                # 单位类型
                type_text = f"类型: {entity.entity_type}"
                text = self.font.render(type_text, True, (200, 200, 200))
                self.screen.blit(text, (panel["rect"].x + 15, info_y))
                info_y += 18
                
                # 血量信息
                if hp:
                    cur_hp = hp.get_param('current_health')
                    max_hp = hp.get_param('max_health')
                    hp_text = f"血量: {cur_hp}/{max_hp}"
                    text = self.font.render(hp_text, True, (200, 200, 200))
                    self.screen.blit(text, (panel["rect"].x + 15, info_y))
                    info_y += 18
                
                # 速度信息
                if mv:
                    speed = mv.get_param('speed')
                    speed_text = f"速度: {speed}"
                    text = self.font.render(speed_text, True, (200, 200, 200))
                    self.screen.blit(text, (panel["rect"].x + 15, info_y))
                    info_y += 18
                
                info_y += 10  # 单位间距
                count += 1
        else:
            no_selection = self.font.render("未选中单位", True, (150, 150, 150))
            self.screen.blit(no_selection, (panel["rect"].x + 15, info_y))

    def draw_bottom_control_bar(self):
        """绘制底部控制栏"""
        bar = self.bottom_control_bar
        # 背景
        pygame.draw.rect(self.screen, bar["bg_color"], bar["rect"])
        pygame.draw.rect(self.screen, bar["border_color"], bar["rect"], 1)
        
        # 时间轴区域（模拟）
        timeline_rect = pygame.Rect(bar["rect"].x + 250, bar["rect"].y + 10, 
                                   bar["rect"].width - 300, 20)
        pygame.draw.rect(self.screen, (40, 60, 100), timeline_rect)
        pygame.draw.rect(self.screen, (80, 100, 140), timeline_rect, 1)
        
        # 时间轴标签
        timeline_label = self.font.render("时间轴", True, (200, 200, 200))
        self.screen.blit(timeline_label, (timeline_rect.x - 50, timeline_rect.y))
        
        # 速度显示
        speed_text = f"速度: x{getattr(self, 'game_speed', 1.0):.1f}"
        speed_label = self.font.render(speed_text, True, (200, 200, 200))
        self.screen.blit(speed_label, (timeline_rect.right + 20, timeline_rect.y))

    def draw_modern_buttons(self):
        """绘制现代化风格的按钮"""
        mouse_pos = pygame.mouse.get_pos()
        
        for btn_list in [self.top_toolbar_buttons, self.right_panel_buttons, self.bottom_control_buttons]:
            for btn in btn_list:
                # 检查鼠标悬停
                is_hovered = btn["rect"].collidepoint(mouse_pos)
                bg_color = btn.get("hover_color", (60, 80, 120)) if is_hovered else btn.get("bg_color", (40, 60, 100))
                
                # 绘制按钮背景
                pygame.draw.rect(self.screen, bg_color, btn["rect"])
                pygame.draw.rect(self.screen, (100, 120, 160), btn["rect"], 1)
                
                # 绘制按钮文字
                text_color = btn.get("text_color", (220, 220, 220))
                text = self.font.render(btn["label"], True, text_color)
                text_rect = text.get_rect(center=btn["rect"].center)
                self.screen.blit(text, text_rect)

    def draw_dropdown_menus(self):
        """绘制下拉菜单"""
        if self.open_menu is not None:
            # 菜单项定义与布局
            items = []
            if self.open_menu == 'view_menu':
                items = [
                    {"label": "叠层开关", "action": "toggle_obstacles"},
                    {"label": "重置视窗", "action": "view_reset"},
                    {"label": "左侧面板", "action": "toggle_left_panel"}
                ]
            elif self.open_menu == 'control_menu':
                items = [
                    {"label": "暂停/继续", "action": "pause_toggle"},
                    {"label": "加速", "action": "speed_up"},
                    {"label": "减速", "action": "speed_down"}
                ]
            elif self.open_menu == 'units_menu':
                items = [
                    {"label": "攻击模式", "action": "attack_mode_toggle"},
                    {"label": "停止", "action": "issue_stop"},
                    {"label": "速度+", "action": "unit_speed_inc"},
                    {"label": "速度-", "action": "unit_speed_dec"},
                    {"label": "左转10°", "action": "heading_left"},
                    {"label": "右转10°", "action": "heading_right"},
                    {"label": "传感器切换", "action": "sensor_toggle"},
                    {"label": "就近攻击", "action": "attack_nearest"}
                ]
            elif self.open_menu == 'help_menu':
                items = [
                    {"label": "快捷键说明", "action": "help_shortcuts"}
                ]

            # 找到对应的顶部按钮
            host_btn = None
            for btn in self.top_toolbar_buttons:
                if btn["action"] == self.open_menu:
                    host_btn = btn
                    break
            
            if host_btn and items:
                menu_w = 160
                item_h = 24
                x = host_btn["rect"].x
                y = host_btn["rect"].bottom + 2
                menu_rect = pygame.Rect(x, y, menu_w, item_h * len(items))
                
                # 菜单背景
                pygame.draw.rect(self.screen, (25, 35, 55), menu_rect)
                pygame.draw.rect(self.screen, (80, 100, 140), menu_rect, 1)
                
                # 绘制每个条目
                for i, item in enumerate(items):
                    item_rect = pygame.Rect(x + 2, y + i*item_h, menu_w - 4, item_h)
                    item["rect"] = item_rect
                    
                    # 悬停效果
                    mouse_pos = pygame.mouse.get_pos()
                    if item_rect.collidepoint(mouse_pos):
                        pygame.draw.rect(self.screen, (50, 70, 100), item_rect)
                    else:
                        pygame.draw.rect(self.screen, (30, 45, 75), item_rect)
                    
                    pygame.draw.rect(self.screen, (80, 100, 140), item_rect, 1)
                    
                    # 文字
                    label = self.font.render(item["label"], True, (220, 220, 220))
                    self.screen.blit(label, (item_rect.x + 6, item_rect.y + 4))
                
                # 缓存菜单项用于事件处理
                self._menu_items_cache = items

        # 帮助浮层
        if self.show_help_overlay:
            panel = pygame.Rect(self.screen_width - 310, self.TOP_BAR_HEIGHT + 10, 300, 160)
            pygame.draw.rect(self.screen, (20, 30, 50), panel)
            pygame.draw.rect(self.screen, (80, 100, 140), panel, 1)
            
            lines = [
                "快捷键说明：",
                "O：叠层开关",
                "方向键：移动视窗", 
                "右键：移动；攻击模式下右键：攻击",
                "按钮/菜单项：对应单位与控制操作"
            ]
            y = panel.y + 10
            for line in lines:
                text = self.font.render(line, True, (220, 220, 220))
                self.screen.blit(text, (panel.x + 10, y))
                y += 25

    def process_ui_action(self, act):
        # 与按钮逻辑一致的动作派发
        if act == "attack_mode_toggle":
            self.attack_mode = not self.attack_mode
        elif act == "issue_stop":
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    StopCommand(entity.entity_id)
        elif act == "unit_speed_inc":
            # 读取当前速度并设置为绝对值（+5）
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    mv = entity.get_component(MovementComponent)
                    cur = mv.get_param('speed') if mv is not None else 0
                    new_speed = max(0, min(100, int(cur) + 5))
                    SetSpeedCommand(entity.entity_id, speed=new_speed)
        elif act == "unit_speed_dec":
            # 读取当前速度并设置为绝对值（-5）
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    mv = entity.get_component(MovementComponent)
                    cur = mv.get_param('speed') if mv is not None else 0
                    new_speed = max(0, min(100, int(cur) - 5))
                    SetSpeedCommand(entity.entity_id, speed=new_speed)
        elif act == "heading_left":
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    RotateHeadingCommand(entity.entity_id, delta_deg=-10)
        elif act == "heading_right":
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    RotateHeadingCommand(entity.entity_id, delta_deg=10)
        elif act == "sensor_toggle":
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    ToggleSensorCommand(entity.entity_id)
        elif act == "attack_nearest":
            for entity in self.game_data.get_all_entities():
                if entity.entity_id in self.selected_ids:
                    AttackNearestCommand(entity.entity_id)
        elif act == "toggle_obstacles":
            self.show_obstacles = not self.show_obstacles
        elif act == "view_reset":
            self.camera_offset = [0, 0]
        elif act == "toggle_left_panel":
            # 切换左侧面板显示
            self.left_info_bar["visible"] = not self.left_info_bar["visible"]
            # 重新计算主视图区域
            left_offset = self.LEFT_PANEL_WIDTH if self.left_info_bar["visible"] else 0
            self.main_view_area = pygame.Rect(
                left_offset, self.TOP_BAR_HEIGHT,
                self.screen_width - self.RIGHT_PANEL_WIDTH - left_offset,
                self.screen_height - self.TOP_BAR_HEIGHT - self.BOTTOM_BAR_HEIGHT
            )
        elif act == "help_shortcuts":
            self.show_help_overlay = not self.show_help_overlay
        else:
            # 其它交由外层处理（暂停/加速/减速等）
            self.ui_actions.append(act)
