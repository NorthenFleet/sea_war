import pygame
import os, sys
from ..ui.font_loader import load_cn_font
from ..core.entities.entity import *
from ..init import Map
from ..ui.player import CommandList, Command, MoveCommand


class RenderManager:
    def __init__(self, env, screen_size, tile_size=64, terrain_override=None):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Sea War')
        self.screen_height = screen_size[1]
        self.screen_width = screen_size[0]
        self.env = env
        self.game_data = env.game_data  # 从环境中获取 game_data
        self.tile_size = tile_size  # 2D 图块的大小
        self.camera_offset = [0, 0]  # 摄像机初始偏移
        self.clock = pygame.time.Clock()
        self.should_close = False
        self.terrain_override = terrain_override  # 指定地图文件名（在 images 目录下）
        self.sprites = self.load_sprites()  # 加载图片
        # 加载地图并计算缩放比例
        self.map = Map('core/data/map.json')
        self.scale_x = self.screen_width / max(1, self.map.global_width)
        self.scale_y = self.screen_height / max(1, self.map.global_height)
        # 使用中文兼容字体
        self.font = load_cn_font(16)

        # 选择与交互状态
        self.selected_ids = set()
        self.entity_screen_rects = {}
        self.selecting = False
        self.selection_start = None
        self.selection_rect = None

        # 指令与UI动作
        self.command_list = CommandList()
        Command.set_command_list(self.command_list)
        self.ui_actions = []  # e.g., ['pause_toggle', 'speed_up']

        # 简易控制按钮
        self.buttons = [
            {"label": "暂停/继续", "action": "pause_toggle", "rect": pygame.Rect(10, 10, 90, 24)},
            {"label": "+ 速度", "action": "speed_up", "rect": pygame.Rect(110, 10, 70, 24)},
            {"label": "- 速度", "action": "speed_down", "rect": pygame.Rect(190, 10, 70, 24)}
        ]
        # 记录所用地图文件名（用于调试显示）
        try:
            self.terrain_filename = os.path.basename(self.sprites.get('terrain_path', ''))
        except Exception:
            self.terrain_filename = None

    def load_sprites(self):
        """加载资源：地形从 render/map 读取，其余单位从 render/images。"""
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        map_dir = os.path.join(os.path.dirname(__file__), 'map')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(map_dir, exist_ok=True)

        # 地图图片候选：如果存在则优先作为地形贴图
        terrain_candidates = [
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
        # 其次使用候选列表（当未选择或不存在时）
        if terrain_path is None:
            for name in terrain_candidates:
                p = os.path.join(map_dir, name)
                if os.path.exists(p):
                    terrain_path = p
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
        # 迷你地图参数
        mini_w, mini_h = 160, 120
        mini_rect = pygame.Rect(self.screen_width - mini_w - 10, self.screen_height - mini_h - 10, mini_w, mini_h)
        pygame.draw.rect(self.screen, (15, 15, 15), mini_rect)
        pygame.draw.rect(self.screen, (80, 80, 80), mini_rect, 1)

        # 绘制地形简化（障碍/地面）
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
                        color = (20, 100, 20) if compressed[by][bx] == 1 else (34, 139, 34)
                        px = int(mini_rect.x + bx * cell_w)
                        py = int(mini_rect.y + by * cell_h)
                        pygame.draw.rect(self.screen, color, (px, py, int(cell_w)+1, int(cell_h)+1))

        # 绘制单位点
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
            color = (220, 30, 30) if owner == 'red' else (30, 30, 220)
            pygame.draw.circle(self.screen, color, (ex, ey), 2)

        # 当前视窗在迷你地图上的矩形
        view_w = int(self.screen_width / max(1, self.scale_x) * mini_scale_x)
        view_h = int(self.screen_height / max(1, self.scale_y) * mini_scale_y)
        view_x = int(mini_rect.x + (self.camera_offset[0] / max(1, self.scale_x)) * mini_scale_x)
        view_y = int(mini_rect.y + (self.camera_offset[1] / max(1, self.scale_y)) * mini_scale_y)
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(view_x, view_y, view_w, view_h), 1)

    def handle_events(self):
        """处理窗口事件，保持窗口响应并允许关闭。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_close = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.should_close = True
                elif event.key == pygame.K_UP:
                    self.camera_offset[1] += 10
                elif event.key == pygame.K_DOWN:
                    self.camera_offset[1] -= 10
                elif event.key == pygame.K_LEFT:
                    self.camera_offset[0] += 10
                elif event.key == pygame.K_RIGHT:
                    self.camera_offset[0] -= 10
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:  # 左键开始框选或点击按钮
                    # 按钮点击检测
                    for b in self.buttons:
                        if b["rect"].collidepoint(mx, my):
                            self.ui_actions.append(b["action"])  # 推入动作队列
                            break
                    else:
                        self.selecting = True
                        self.selection_start = (mx, my)
                        self.selection_rect = pygame.Rect(mx, my, 0, 0)
                elif event.button == 3:  # 右键移动
                    # 将鼠标位置转为地图坐标
                    target_x = int((mx - self.camera_offset[0]) / max(1e-6, self.scale_x))
                    target_y = int((my - self.camera_offset[1]) / max(1e-6, self.scale_y))
                    for entity in self.game_data.get_all_entities():
                        if entity.entity_id in self.selected_ids:
                            mv = entity.get_component(MovementComponent)
                            speed = mv.get_param('speed') if mv else 1
                            MoveCommand(entity.entity_id, target_position=(target_x, target_y), speed=speed)
            elif event.type == pygame.MOUSEMOTION:
                if self.selecting and self.selection_start:
                    sx, sy = self.selection_start
                    mx, my = event.pos
                    x = min(sx, mx)
                    y = min(sy, my)
                    w = abs(mx - sx)
                    h = abs(my - sy)
                    self.selection_rect = pygame.Rect(x, y, w, h)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.selecting:
                    self.selecting = False
                    # 选择实体（框选或点选）
                    if self.selection_rect and (self.selection_rect.width > 3 and self.selection_rect.height > 3):
                        self.selected_ids = set()
                        for eid, rect in self.entity_screen_rects.items():
                            if self.selection_rect.colliderect(rect):
                                self.selected_ids.add(eid)
                    else:
                        # 点选：选择鼠标位置下的第一个实体
                        mx, my = event.pos
                        clicked = None
                        for eid, rect in self.entity_screen_rects.items():
                            if rect.collidepoint(mx, my):
                                clicked = eid
                                break
                        self.selected_ids = {clicked} if clicked else set()
                    self.selection_rect = None

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
        """主渲染更新函数"""
        # 事件泵，确保窗口出现并响应
        self.handle_events()
        if self.should_close:
            # 结束渲染循环，让外层游戏逻辑退出
            pygame.display.quit()
            return False
        # 清空屏幕
        self.screen.fill((0, 0, 0))  # 黑色背景

        # 1. 绘制地形
        self.draw_terrain()

        # 2. 绘制单位
        self.draw_units()

        # 3. 绘制UI按钮
        self.draw_buttons()

        # 4. 绘制小地图
        self.draw_minimap()

        # 5. 绘制选中单位状态面板
        self.draw_status_panel()

        # 5.1 在屏幕右下角显示当前地图文件名
        if self.sprites.get('terrain_path'):
            name = os.path.basename(self.sprites['terrain_path'])
            label = self.font.render(f"地图: {name}", True, (255, 255, 255))
            self.screen.blit(label, (self.screen_width - label.get_width() - 12, self.screen_height - label.get_height() - 12))

        # 6. 刷新显示
        pygame.display.flip()

        # 控制帧率
        self.clock.tick(60)
        return True

    def draw_status_panel(self):
        panel_rect = pygame.Rect(10, 44, 260, 140)
        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), panel_rect, 1)
        title = self.font.render("选中单位", True, (255, 255, 255))
        self.screen.blit(title, (panel_rect.x + 6, panel_rect.y + 6))

        # 列出部分选中单位的状态
        y = panel_rect.y + 24
        count = 0
        for entity in self.game_data.get_all_entities():
            if entity.entity_id not in self.selected_ids:
                continue
            pos = entity.get_component(PositionComponent)
            mv = entity.get_component(MovementComponent)
            hp = entity.get_component(HealthComponent)
            owner = self.game_data.get_unit_owner(entity.entity_id)
            t = str(entity.entity_type)
            h = hp.get_param('current_health') if hp else None
            s = mv.get_param('speed') if mv else None
            line = f"[{owner}] {t}  HP:{h if h is not None else '-'}  SPD:{s if s is not None else '-'}"
            text = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(text, (panel_rect.x + 6, y))
            y += 18
            count += 1
            if count >= 6:
                break
