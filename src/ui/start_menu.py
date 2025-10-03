import os
import pygame
from .font_loader import load_cn_font


class StartMenu:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font_big = load_cn_font(48)
        self.font_small = load_cn_font(22)
        # 搜索可选地图文件
        self.images_dir = os.path.join(os.path.dirname(__file__), '..', 'render', 'images')
        self.images_dir = os.path.abspath(self.images_dir)
        self.candidates = self._find_candidates()
        self.selected_index = 0 if self.candidates else -1

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
        return names

    def run(self, screen_size=(1280, 800), auto_select_timeout=None):
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Sea War - 启动菜单')
        selected_name = None
        start_ms = pygame.time.get_ticks()

        while True:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return None
                        if self.candidates:
                            if event.key in (pygame.K_UP, pygame.K_w):
                                self.selected_index = max(0, self.selected_index - 1)
                            elif event.key in (pygame.K_DOWN, pygame.K_s):
                                self.selected_index = min(len(self.candidates) - 1, self.selected_index + 1)
                            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                                selected_name = self.candidates[self.selected_index]
                                return selected_name

                # 自动选择（用于自动化验证）
                if auto_select_timeout is not None and self.candidates:
                    elapsed = (pygame.time.get_ticks() - start_ms) / 1000.0
                    if elapsed >= auto_select_timeout:
                        return self.candidates[self.selected_index]

                screen.fill((20, 30, 40))
                title = self.font_big.render('海战模拟 - 启动菜单', True, (230, 230, 230))
                screen.blit(title, (screen_size[0]//2 - title.get_width()//2, 80))

                hint = self.font_small.render('方向键选择地图，回车开始；Esc跳过使用默认', True, (200, 200, 200))
                screen.blit(hint, (screen_size[0]//2 - hint.get_width()//2, 140))

                # 列表显示候选地图文件
                start_y = 200
                item_h = 28
                for i, name in enumerate(self.candidates[:10]):
                    color = (255, 255, 255) if i == self.selected_index else (180, 180, 180)
                    label = self.font_small.render(name, True, color)
                    screen.blit(label, (screen_size[0]//2 - 240, start_y + i * item_h))
                    if i == self.selected_index:
                        pygame.draw.rect(screen, (120, 180, 255), (screen_size[0]//2 - 260, start_y + i * item_h - 2, 520, item_h + 4), 1)

                # 预览选中地图（右侧缩略图）
                if self.candidates and 0 <= self.selected_index < len(self.candidates):
                    try:
                        path = os.path.join(self.images_dir, self.candidates[self.selected_index])
                        image = pygame.image.load(path)
                        preview_w, preview_h = 360, 220
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
                return None

        return selected_name