#!/usr/bin/env python3
"""
测试视觉效果系统
"""

import pygame
import sys
import os
import time
import random
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.render.layered_renderer import LayeredRenderManager, ExplosionEffect, TrailEffect, RenderLayer
from src.render.visual_effects import VisualEffectsManager, ExplosionEffect as VFXExplosion, TrailEffect as VFXTrail, MuzzleFlashEffect
from src.render.sprite_manager import LayeredSpriteManager
from src.render.optimized_loader import OptimizedSpriteLoader

class VisualEffectsTest:
    def __init__(self):
        pygame.init()
        self.screen_size = (1200, 800)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("视觉效果测试")
        self.clock = pygame.time.Clock()
        
        # 初始化渲染系统
        self.layered_renderer = LayeredRenderManager(self.screen)
        self.sprite_loader = OptimizedSpriteLoader()
        self.sprite_manager = LayeredSpriteManager(self.sprite_loader)
        
        # 初始化视觉效果管理器
        self.visual_effects = VisualEffectsManager()
        
        # 测试计时器
        self.last_explosion_time = 0
        self.last_trail_time = 0
        self.last_muzzle_flash_time = 0
        
        # 字体
        self.font = pygame.font.Font(None, 36)
        
        print("视觉效果测试初始化完成")
        print("控制说明:")
        print("- 鼠标左键: 添加爆炸效果")
        print("- 鼠标右键: 添加轨迹效果")
        print("- 空格键: 添加枪口闪光效果")
        print("- ESC: 退出")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # 添加枪口闪光效果
                    pos = pygame.mouse.get_pos()
                    self.add_muzzle_flash(pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if event.button == 1:  # 左键 - 爆炸效果
                    self.add_explosion(pos)
                elif event.button == 3:  # 右键 - 轨迹效果
                    self.add_trail(pos)
        return True
    
    def add_explosion(self, pos):
        """添加爆炸效果"""
        # 使用LayeredRenderManager的爆炸效果
        size = random.uniform(30, 80)
        color = random.choice([(255, 100, 0), (255, 150, 50), (255, 200, 100)])
        explosion = ExplosionEffect(pos, size, color, duration=random.uniform(0.5, 1.5))
        self.layered_renderer.add_effect(explosion)
        
        # 同时使用VisualEffectsManager的爆炸效果
        self.visual_effects.add_explosion(pos, size)
        
        print(f"添加爆炸效果在位置 {pos}, 大小: {size:.1f}")
    
    def add_trail(self, end_pos):
        """添加轨迹效果"""
        # 随机起始位置
        start_pos = (
            random.randint(0, self.screen_size[0]),
            random.randint(0, self.screen_size[1])
        )
        
        # 使用LayeredRenderManager的轨迹效果
        width = random.randint(2, 6)
        color = random.choice([(255, 255, 255), (0, 255, 255), (255, 255, 0)])
        trail = TrailEffect(start_pos, end_pos, width, color, duration=random.uniform(1.0, 2.0))
        self.layered_renderer.add_effect(trail)
        
        # 同时使用VisualEffectsManager的轨迹效果
        self.visual_effects.add_trail(start_pos, end_pos, color)
        
        print(f"添加轨迹效果从 {start_pos} 到 {end_pos}")
    
    def add_muzzle_flash(self, pos):
        """添加枪口闪光效果"""
        # 使用VisualEffectsManager的枪口闪光效果
        direction = random.uniform(0, 2 * math.pi)
        self.visual_effects.add_muzzle_flash(pos, direction)
        
        print(f"添加枪口闪光效果在位置 {pos}")
    
    def add_automatic_effects(self):
        """自动添加一些效果进行演示"""
        current_time = time.time()
        
        # 每2秒添加一个随机爆炸
        if current_time - self.last_explosion_time > 2.0:
            pos = (
                random.randint(100, self.screen_size[0] - 100),
                random.randint(100, self.screen_size[1] - 100)
            )
            self.add_explosion(pos)
            self.last_explosion_time = current_time
        
        # 每1.5秒添加一个随机轨迹
        if current_time - self.last_trail_time > 1.5:
            end_pos = (
                random.randint(50, self.screen_size[0] - 50),
                random.randint(50, self.screen_size[1] - 50)
            )
            self.add_trail(end_pos)
            self.last_trail_time = current_time
        
        # 每0.5秒添加一个枪口闪光
        if current_time - self.last_muzzle_flash_time > 0.5:
            pos = (
                random.randint(50, self.screen_size[0] - 50),
                random.randint(50, self.screen_size[1] - 50)
            )
            self.add_muzzle_flash(pos)
            self.last_muzzle_flash_time = current_time
    
    def render_info(self):
        """渲染信息文本"""
        info_texts = [
            "视觉效果测试",
            f"LayeredRenderer 效果数量: {len(self.layered_renderer.effects)}",
            f"VisualEffects 效果数量: {len(self.visual_effects.effects)}",
            "",
            "控制:",
            "鼠标左键 - 爆炸效果",
            "鼠标右键 - 轨迹效果", 
            "空格键 - 枪口闪光",
            "ESC - 退出"
        ]
        
        y_offset = 10
        for text in info_texts:
            if text:  # 跳过空行
                surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(surface, (10, y_offset))
            y_offset += 30
    
    def run(self):
        """运行测试"""
        running = True
        
        while running:
            delta_time = self.clock.tick(60) / 1000.0
            
            # 处理事件
            running = self.handle_events()
            
            # 自动添加效果
            self.add_automatic_effects()
            
            # 清屏
            self.screen.fill((20, 30, 50))  # 深蓝色背景
            
            # 更新和渲染LayeredRenderer效果
            self.layered_renderer.render(self.screen, dt=delta_time)
            
            # 更新和渲染VisualEffects效果
            self.visual_effects.update(delta_time)
            self.visual_effects.render(self.screen, (0, 0))
            
            # 渲染信息
            self.render_info()
            
            # 更新显示
            pygame.display.flip()
        
        pygame.quit()
        print("视觉效果测试结束")

if __name__ == "__main__":
    test = VisualEffectsTest()
    test.run()