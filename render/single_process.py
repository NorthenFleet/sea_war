import pygame
import math
from component import PositionComponent


class RenderManager:
    def __init__(self, env, screen_size=(800, 600), tile_size=64):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        self.env = env
        self.game_data = env.game_data  # 从环境中获取 game_data
        self.tile_size = tile_size  # 2D 图块的大小
        self.camera_offset = [0, 0]  # 摄像机初始偏移
        self.clock = pygame.time.Clock()
        self.sprites = self.load_sprites()  # 加载精灵图

    def load_sprites(self):
        """加载游戏中的精灵图，作为2.5D显示基础"""
        sprites = {}
        # 示例：加载地形和单位的图片
        sprites['terrain'] = pygame.image.load('assets/terrain.png')
        sprites['unit'] = pygame.image.load('assets/unit.png')
        return sprites

    def isometric_transform(self, x, y):
        """将 2D 坐标转换为等轴测 2.5D 坐标"""
        iso_x = (x - y) * (self.tile_size // 2)
        iso_y = (x + y) * (self.tile_size // 4)
        return iso_x, iso_y

    def draw_terrain(self):
        """绘制地形图层"""
        terrain_grid = self.env.map.grid  # 假设 map 包含地形网格信息
        for row in range(len(terrain_grid)):
            for col in range(len(terrain_grid[row])):
                tile = terrain_grid[row][col]
                iso_x, iso_y = self.isometric_transform(row, col)
                self.screen.blit(self.sprites['terrain'],
                                 (iso_x + self.camera_offset[0],
                                  iso_y + self.camera_offset[1]))

    def draw_units(self):
        """绘制单位图层"""
        for entity in self.game_data.get_all_entities():
            position = entity.get_component(PositionComponent)
            if position:
                iso_x, iso_y = self.isometric_transform(position.x, position.y)
                self.screen.blit(self.sprites['unit'],
                                 (iso_x + self.camera_offset[0],
                                  iso_y + self.camera_offset[1] - 16))  # 偏移用于调整显示

    def update(self):
        """主渲染更新函数"""
        # 清空屏幕
        self.screen.fill((0, 0, 0))  # 黑色背景

        # 1. 绘制地形
        self.draw_terrain()

        # 2. 绘制单位
        self.draw_units()

        # 3. 刷新显示
        pygame.display.flip()

        # 控制帧率
        self.clock.tick(60)
