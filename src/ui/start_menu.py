import os
import pygame
from .font_loader import load_cn_font


class StartMenu:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font_big = load_cn_font(48)
        self.font_small = load_cn_font(22)
        # 搜索可选地图文件（已迁移到 render/map）
        self.images_dir = os.path.join(os.path.dirname(__file__), '..', 'render', 'map')
        self.images_dir = os.path.abspath(self.images_dir)
        self.candidates = self._find_candidates()
        self.selected_index = 0 if self.candidates else -1
        # 额外资源目录
        self.scenario_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'data'))
        self.saves_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'saves'))
        os.makedirs(self.saves_dir, exist_ok=True)

    def _find_candidates(self):
        if not os.path.exists(self.images_dir):
            return []
        allow_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
        names = []
        for f in os.listdir(self.images_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in allow_ext:
                names.append(f)
        # 优先常用命名靠前
        priority = ['地图.png', 'map.png', 'terrain.png']
        names.sort(key=lambda n: (0 if n in priority else 1, n))
        # 追加内置纯色背景预设，作为可选“地图”
        color_presets = [
            'color:#1E90FF',  # 深海蓝
            'color:#2E8B57',  # 海绿色
            'color:#87CEEB',  # 天空蓝
            'color:#808080',  # 中性灰
            'color:#000000',  # 纯黑
        ]
        for c in color_presets:
            if c not in names:
                names.append(c)
        return names

    def _parse_color_value(self, value):
        """解析字符串形式的颜色，如 color:#RRGGBB 或 color:R,G,B。返回 (r,g,b) 或 None。"""
        try:
            if not isinstance(value, str):
                return None
            if not value.lower().startswith('color:'):
                return None
            raw = value.split(':', 1)[1].strip()
            if raw.startswith('#') and len(raw) == 7:
                r = int(raw[1:3], 16)
                g = int(raw[3:5], 16)
                b = int(raw[5:7], 16)
                return (r, g, b)
            # 逗号分隔的数字
            parts = raw.split(',')
            if len(parts) == 3:
                r, g, b = [max(0, min(255, int(p.strip()))) for p in parts]
                return (r, g, b)
        except Exception:
            return None
        return None

    def run(self, screen_size=(1280, 800), auto_select_timeout=None):
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Sea War - 启动菜单')
        selected_name = None
        start_ms = pygame.time.get_ticks()
        # 开始按钮区域（底部居中）
        btn_w, btn_h = 220, 52
        btn_rect = pygame.Rect(screen_size[0]//2 - btn_w//2, screen_size[1] - 140, btn_w, btn_h)
        # 扩展操作按钮（左侧）
        side_btn_w, side_btn_h = 200, 36
        side_btns = {
            'start': pygame.Rect(40, 220, side_btn_w, side_btn_h),
            'open_scenario': pygame.Rect(40, 270, side_btn_w, side_btn_h),
            'load_save': pygame.Rect(40, 320, side_btn_w, side_btn_h),
            'exit': pygame.Rect(40, 370, side_btn_w, side_btn_h),
        }
        # 当前列表模式：map 或 scenario 或 saves
        list_mode = 'map'
        scenario_files = self._find_scenarios()
        save_files = self._find_saves()
        list_selected_index = 0

        # 列表区域点击判定参数
        list_start_y = 200
        item_h = 28
        list_left_x = screen_size[0]//2 - 260
        list_width = 520

        while True:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        try:
                            pygame.display.quit()
                            pygame.event.clear()
                            pygame.quit()
                        except Exception:
                            pass
                        return None
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        # 根据当前模式，点击列表区域可选择条目；双击可直接开始/打开
                        list_height = 0
                        items = []
                        if list_mode == 'map':
                            items = self.candidates[:10]
                        elif list_mode == 'scenario':
                            items = scenario_files[:10]
                        elif list_mode == 'saves':
                            items = save_files[:10]
                        list_height = item_h * len(items)
                        list_rect = pygame.Rect(list_left_x, list_start_y, list_width, list_height)
                        if list_rect.collidepoint(mx, my):
                            idx = (my - list_start_y) // item_h
                            if 0 <= idx < len(items):
                                if list_mode == 'map':
                                    self.selected_index = idx
                                else:
                                    list_selected_index = idx
                        # 点击主按钮区域同原逻辑
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            try:
                                pygame.display.quit()
                                pygame.event.clear()
                                pygame.quit()
                            except Exception:
                                pass
                            return None
                        if self.candidates:
                            if event.key in (pygame.K_UP, pygame.K_w):
                                self.selected_index = max(0, self.selected_index - 1)
                            elif event.key in (pygame.K_DOWN, pygame.K_s):
                                self.selected_index = min(len(self.candidates) - 1, self.selected_index + 1)
                            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                                # 键盘触发“开始游戏”
                                selected_name = self.candidates[self.selected_index]
                                try:
                                    pygame.display.quit()
                                    pygame.event.clear()
                                    pygame.quit()
                                except Exception:
                                    pass
                                return selected_name
                        # 快捷键切换模式
                        if event.key == pygame.K_1:
                            list_mode = 'map'
                        elif event.key == pygame.K_2:
                            list_mode = 'scenario'
                        elif event.key == pygame.K_3:
                            list_mode = 'saves'

                # 鼠标点击开始按钮（仅在按钮可用时响应）
                if pygame.mouse.get_pressed()[0]:
                    mx, my = pygame.mouse.get_pos()
                    if btn_rect.collidepoint(mx, my):
                        if self.candidates and 0 <= self.selected_index < len(self.candidates):
                            selected_name = self.candidates[self.selected_index]
                            try:
                                pygame.display.quit()
                                pygame.event.clear()
                                pygame.quit()
                            except Exception:
                                pass
                            return selected_name
                        else:
                            # 未选中或无候选，忽略点击，保持菜单显示
                            pass
                    # 扩展操作按钮
                    for key, rect in side_btns.items():
                        if rect.collidepoint(mx, my):
                            if key == 'start':
                                # 与“开始游戏”一致
                                if self.candidates and 0 <= self.selected_index < len(self.candidates):
                                    selected_name = self.candidates[self.selected_index]
                                    try:
                                        pygame.display.quit()
                                        pygame.event.clear()
                                        pygame.quit()
                                    except Exception:
                                        pass
                                    return selected_name
                            elif key == 'open_scenario':
                                list_mode = 'scenario'
                            elif key == 'load_save':
                                list_mode = 'saves'
                            elif key == 'exit':
                                try:
                                    pygame.display.quit()
                                    pygame.event.clear()
                                    pygame.quit()
                                except Exception:
                                    pass
                                return None

                # 自动选择（用于自动化验证；默认不启用）
                if auto_select_timeout is not None:
                    elapsed = (pygame.time.get_ticks() - start_ms) / 1000.0
                    if elapsed >= auto_select_timeout:
                        # 仅在自动化场景下允许自动开始
                        if self.candidates and 0 <= self.selected_index < len(self.candidates):
                            try:
                                pygame.display.quit()
                                pygame.event.clear()
                                pygame.quit()
                            except Exception:
                                pass
                            return self.candidates[self.selected_index]
                        else:
                            try:
                                pygame.display.quit()
                                pygame.event.clear()
                                pygame.quit()
                            except Exception:
                                pass
                            return None

                screen.fill((20, 30, 40))
                title = self.font_big.render('海战模拟 - 启动菜单', True, (230, 230, 230))
                screen.blit(title, (screen_size[0]//2 - title.get_width()//2, 80))

                hint = self.font_small.render('方向键或鼠标点击列表选择；点击下方“开始游戏”进入', True, (200, 200, 200))
                screen.blit(hint, (screen_size[0]//2 - hint.get_width()//2, 140))

                # 绘制“开始游戏”按钮
                enabled = self.candidates and 0 <= self.selected_index < len(self.candidates)
                btn_color = (60, 160, 80) if enabled else (90, 90, 90)
                border_color = (200, 200, 200)
                pygame.draw.rect(screen, btn_color, btn_rect, border_radius=8)
                pygame.draw.rect(screen, border_color, btn_rect, width=2, border_radius=8)
                btn_text = self.font_small.render('开始游戏', True, (255, 255, 255))
                screen.blit(btn_text, (btn_rect.centerx - btn_text.get_width()//2, btn_rect.centery - btn_text.get_height()//2))
                if not enabled:
                    tip = self.font_small.render('请选择地图后再开始', True, (220, 180, 100))
                    screen.blit(tip, (btn_rect.centerx - tip.get_width()//2, btn_rect.bottom + 8))

                # 左侧扩展操作按钮
                for key, rect in side_btns.items():
                    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=6)
                    pygame.draw.rect(screen, (180, 180, 180), rect, 1, border_radius=6)
                    text = {
                        'start': '开始',
                        'open_scenario': '打开想定',
                        'load_save': '读取存档',
                        'exit': '退出'
                    }[key]
                    label = self.font_small.render(text, True, (240, 240, 240))
                    screen.blit(label, (rect.x + rect.w//2 - label.get_width()//2, rect.y + rect.h//2 - label.get_height()//2))

                # 列表显示（根据模式：地图/想定/存档）
                start_y = 200
                item_h = 28
                if list_mode == 'map':
                    for i, name in enumerate(self.candidates[:10]):
                        color = (255, 255, 255) if i == self.selected_index else (180, 180, 180)
                        label = self.font_small.render(name, True, color)
                        screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                        if i == self.selected_index:
                            pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)
                elif list_mode == 'scenario':
                    for i, name in enumerate(scenario_files[:10]):
                        color = (255, 255, 255) if i == list_selected_index else (180, 180, 180)
                        label = self.font_small.render(name, True, color)
                        screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                        if i == list_selected_index:
                            pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)
                elif list_mode == 'saves':
                    for i, name in enumerate(save_files[:10]):
                        color = (255, 255, 255) if i == list_selected_index else (180, 180, 180)
                        label = self.font_small.render(name, True, color)
                        screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                        if i == list_selected_index:
                            pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)

                # 预览选中地图（右侧缩略图）
                if self.candidates and 0 <= self.selected_index < len(self.candidates):
                    try:
                        cand = self.candidates[self.selected_index]
                        preview_w, preview_h = 360, 220
                        color = self._parse_color_value(cand)
                        if color is not None:
                            surf = pygame.Surface((preview_w, preview_h))
                            surf.fill(color)
                            screen.blit(surf, (screen_size[0]//2 + 140, 200))
                        else:
                            path = os.path.join(self.images_dir, cand)
                            image = pygame.image.load(path)
                            image = pygame.transform.smoothscale(image, (preview_w, preview_h))
                            screen.blit(image, (screen_size[0]//2 + 140, 200))
                        border = pygame.Rect(screen_size[0]//2 + 140, 200, preview_w, preview_h)
                        pygame.draw.rect(screen, (180, 180, 180), border, 1)
                    except Exception:
                        pass

                pygame.display.flip()
                self.clock.tick(60)

            except KeyboardInterrupt:
                # 优雅中断
                try:
                    pygame.display.quit()
                    pygame.event.clear()
                    pygame.quit()
                except Exception:
                    pass
                return None

        return selected_name

    def run_extended(self, screen_size=(1280, 800), auto_select_timeout=None):
        """扩展模式：返回结构化菜单选择。
        返回示例：
          {'action': 'start', 'terrain': 'ground.png'}
          {'action': 'open_scenario', 'scenario': 'air_defense.json'}
          {'action': 'load_save', 'save': 'save_001.json'}
          {'action': 'exit'}
        兼容 auto_select_timeout 用于自动测试。
        """
        # 基于原 run 渲染，但支持左侧按钮点击返回结构化结果
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Sea War - 启动菜单（扩展）')
        start_ms = pygame.time.get_ticks()
        btn_w, btn_h = 220, 52
        btn_rect = pygame.Rect(screen_size[0]//2 - btn_w//2, screen_size[1] - 140, btn_w, btn_h)

        side_btn_w, side_btn_h = 200, 36
        side_btns = {
            'start': pygame.Rect(40, 220, side_btn_w, side_btn_h),
            'open_scenario': pygame.Rect(40, 270, side_btn_w, side_btn_h),
            'load_save': pygame.Rect(40, 320, side_btn_w, side_btn_h),
            'exit': pygame.Rect(40, 370, side_btn_w, side_btn_h),
        }
        list_mode = 'map'
        scenario_files = self._find_scenarios()
        save_files = self._find_saves()
        sel_map = 0 if self.candidates else -1
        sel_other = 0

        while True:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        try:
                            pygame.display.quit(); pygame.event.clear(); pygame.quit()
                        except Exception:
                            pass
                        return {'action': 'exit'}
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        # 列表区域点击选择索引
                        list_start_y = 200
                        item_h = 28
                        list_left_x = screen_size[0]//2 - 260
                        list_width = 520
                        items = []
                        if list_mode == 'map':
                            items = self.candidates[:10]
                        elif list_mode == 'scenario':
                            items = scenario_files[:10]
                        elif list_mode == 'saves':
                            items = save_files[:10]
                        list_rect = pygame.Rect(list_left_x, list_start_y, list_width, item_h * len(items))
                        if list_rect.collidepoint(mx, my):
                            idx = (my - list_start_y) // item_h
                            if 0 <= idx < len(items):
                                if list_mode == 'map':
                                    sel_map = idx
                                else:
                                    sel_other = idx
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            try:
                                pygame.display.quit(); pygame.event.clear(); pygame.quit()
                            except Exception:
                                pass
                            return {'action': 'exit'}
                        if list_mode == 'map' and self.candidates:
                            if event.key in (pygame.K_UP, pygame.K_w):
                                sel_map = max(0, sel_map - 1)
                            elif event.key in (pygame.K_DOWN, pygame.K_s):
                                sel_map = min(len(self.candidates) - 1, sel_map + 1)
                            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                                if sel_map >= 0:
                                    name = self.candidates[sel_map]
                                    try:
                                        pygame.display.quit(); pygame.event.clear(); pygame.quit()
                                    except Exception:
                                        pass
                                    return {'action': 'start', 'terrain': name}
                        elif list_mode == 'scenario' and scenario_files:
                            if event.key in (pygame.K_UP, pygame.K_w):
                                sel_other = max(0, sel_other - 1)
                            elif event.key in (pygame.K_DOWN, pygame.K_s):
                                sel_other = min(len(scenario_files) - 1, sel_other + 1)
                            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                                name = scenario_files[sel_other]
                                try:
                                    pygame.display.quit(); pygame.event.clear(); pygame.quit()
                                except Exception:
                                    pass
                                return {'action': 'open_scenario', 'scenario': name}
                        elif list_mode == 'saves' and save_files:
                            if event.key in (pygame.K_UP, pygame.K_w):
                                sel_other = max(0, sel_other - 1)
                            elif event.key in (pygame.K_DOWN, pygame.K_s):
                                sel_other = min(len(save_files) - 1, sel_other + 1)
                            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                                name = save_files[sel_other]
                                try:
                                    pygame.display.quit(); pygame.event.clear(); pygame.quit()
                                except Exception:
                                    pass
                                return {'action': 'load_save', 'save': name}
                        if event.key == pygame.K_1:
                            list_mode = 'map'
                        elif event.key == pygame.K_2:
                            list_mode = 'scenario'
                        elif event.key == pygame.K_3:
                            list_mode = 'saves'

                if pygame.mouse.get_pressed()[0]:
                    mx, my = pygame.mouse.get_pos()
                    if btn_rect.collidepoint(mx, my) and sel_map >= 0:
                        name = self.candidates[sel_map]
                        try:
                            pygame.display.quit(); pygame.event.clear(); pygame.quit()
                        except Exception:
                            pass
                        return {'action': 'start', 'terrain': name}
                    for key, rect in side_btns.items():
                        if rect.collidepoint(mx, my):
                            if key == 'start' and sel_map >= 0:
                                name = self.candidates[sel_map]
                                try:
                                    pygame.display.quit(); pygame.event.clear(); pygame.quit()
                                except Exception:
                                    pass
                                return {'action': 'start', 'terrain': name}
                            elif key == 'open_scenario':
                                list_mode = 'scenario'
                            elif key == 'load_save':
                                list_mode = 'saves'
                            elif key == 'exit':
                                try:
                                    pygame.display.quit(); pygame.event.clear(); pygame.quit()
                                except Exception:
                                    pass
                                return {'action': 'exit'}

                # 自动选择（用于自动化验证）
                if auto_select_timeout is not None:
                    elapsed = (pygame.time.get_ticks() - start_ms) / 1000.0
                    if elapsed >= auto_select_timeout:
                        if sel_map >= 0:
                            try:
                                pygame.display.quit(); pygame.event.clear(); pygame.quit()
                            except Exception:
                                pass
                            return {'action': 'start', 'terrain': self.candidates[sel_map]}
                        else:
                            try:
                                pygame.display.quit(); pygame.event.clear(); pygame.quit()
                            except Exception:
                                pass
                            return {'action': 'exit'}

                # 绘制背景与标题
                screen.fill((20, 30, 40))
                title = self.font_big.render('海战模拟 - 启动菜单（扩展）', True, (230, 230, 230))
                screen.blit(title, (screen_size[0]//2 - title.get_width()//2, 80))
                hint = self.font_small.render('1 地图  2 想定  3 存档  回车选择', True, (200, 200, 200))
                # 鼠标也可点击列表选择；下方按钮可开始/打开
                screen.blit(hint, (screen_size[0]//2 - hint.get_width()//2, 140))

                # 主按钮与左侧按钮
                enabled = sel_map >= 0
                btn_color = (60, 160, 80) if enabled else (90, 90, 90)
                border_color = (200, 200, 200)
                pygame.draw.rect(screen, btn_color, btn_rect, border_radius=8)
                pygame.draw.rect(screen, border_color, btn_rect, width=2, border_radius=8)
                btn_text = self.font_small.render('开始游戏', True, (255, 255, 255))
                screen.blit(btn_text, (btn_rect.centerx - btn_text.get_width()//2, btn_rect.centery - btn_text.get_height()//2))

                side_btn_w, side_btn_h = 200, 36
                side_btns = {
                    'start': pygame.Rect(40, 220, side_btn_w, side_btn_h),
                    'open_scenario': pygame.Rect(40, 270, side_btn_w, side_btn_h),
                    'load_save': pygame.Rect(40, 320, side_btn_w, side_btn_h),
                    'exit': pygame.Rect(40, 370, side_btn_w, side_btn_h),
                }
                for key, rect in side_btns.items():
                    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=6)
                    pygame.draw.rect(screen, (180, 180, 180), rect, 1, border_radius=6)
                    text = {
                        'start': '开始',
                        'open_scenario': '打开想定',
                        'load_save': '读取存档',
                        'exit': '退出'
                    }[key]
                    label = self.font_small.render(text, True, (240, 240, 240))
                    screen.blit(label, (rect.x + rect.w//2 - label.get_width()//2, rect.y + rect.h//2 - label.get_height()//2))

                # 右侧列表：根据模式展示文件名
                start_y = 200
                item_h = 28
                if list_mode == 'map':
                    for i, name in enumerate(self.candidates[:10]):
                        color = (255, 255, 255) if i == sel_map else (180, 180, 180)
                        label = self.font_small.render(name, True, color)
                        screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                        if i == sel_map:
                            pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)
                elif list_mode == 'scenario':
                    for i, name in enumerate(scenario_files[:10]):
                        color = (255, 255, 255) if i == sel_other else (180, 180, 180)
                        label = self.font_small.render(name, True, color)
                        screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                        if i == sel_other:
                            pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)
                elif list_mode == 'saves':
                    for i, name in enumerate(save_files[:10]):
                        color = (255, 255, 255) if i == sel_other else (180, 180, 180)
                        label = self.font_small.render(name, True, color)
                        screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                        if i == sel_other:
                            pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)

                # 地图预览
                if list_mode == 'map' and sel_map >= 0:
                    try:
                        cand = self.candidates[sel_map]
                        preview_w, preview_h = 360, 220
                        color = self._parse_color_value(cand)
                        if color is not None:
                            surf = pygame.Surface((preview_w, preview_h))
                            surf.fill(color)
                            screen.blit(surf, (screen_size[0]//2 + 140, 200))
                        else:
                            path = os.path.join(self.images_dir, cand)
                            image = pygame.image.load(path)
                            image = pygame.transform.smoothscale(image, (preview_w, preview_h))
                            screen.blit(image, (screen_size[0]//2 + 140, 200))
                        border = pygame.Rect(screen_size[0]//2 + 140, 200, preview_w, preview_h)
                        pygame.draw.rect(screen, (180, 180, 180), border, 1)
                    except Exception:
                        pass

                pygame.display.flip()
                self.clock.tick(60)
            except KeyboardInterrupt:
                try:
                    pygame.display.quit(); pygame.event.clear(); pygame.quit()
                except Exception:
                    pass
                return {'action': 'exit'}

    def _find_scenarios(self):
        if not os.path.isdir(self.scenario_dir):
            return []
        names = []
        for f in os.listdir(self.scenario_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext == '.json':
                names.append(f)
        names.sort()
        return names

    def _find_saves(self):
        if not os.path.isdir(self.saves_dir):
            return []
        names = []
        for f in os.listdir(self.saves_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext == '.json':
                names.append(f)
        names.sort()
        return names