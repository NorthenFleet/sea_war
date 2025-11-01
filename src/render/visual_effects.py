"""
视觉效果管理器
处理游戏中的各种视觉效果，如爆炸、轨迹、粒子效果等
"""

import pygame
import math
import random
from typing import List, Tuple, Dict, Any


class VisualEffect:
    """基础视觉效果类"""
    
    def __init__(self, position: Tuple[float, float], duration: float = 1.0):
        self.position = position
        self.duration = duration
        self.elapsed_time = 0.0
        self.active = True
    
    def update(self, dt: float):
        """更新效果"""
        self.elapsed_time += dt
        if self.elapsed_time >= self.duration:
            self.active = False
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """渲染效果"""
        pass
    
    def is_finished(self) -> bool:
        """检查效果是否结束"""
        return not self.active


class ExplosionEffect(VisualEffect):
    """爆炸效果"""
    
    def __init__(self, position: Tuple[float, float], size: float = 50.0, duration: float = 0.8):
        super().__init__(position, duration)
        self.size = size
        self.max_size = size
        self.particles = []
        
        # 创建粒子
        particle_count = int(size / 5)
        for _ in range(particle_count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 80)
            particle = {
                'x': position[0],
                'y': position[1],
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.uniform(0.3, 0.8),
                'max_life': random.uniform(0.3, 0.8),
                'size': random.uniform(2, 6)
            }
            self.particles.append(particle)
    
    def update(self, dt: float):
        super().update(dt)
        
        # 更新粒子
        for particle in self.particles[:]:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['life'] -= dt
            
            # 重力效果
            particle['vy'] += 50 * dt
            
            # 空气阻力
            particle['vx'] *= 0.98
            particle['vy'] *= 0.98
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        if not self.active:
            return
        
        # 渲染主爆炸圆圈
        progress = self.elapsed_time / self.duration
        current_size = self.max_size * (1.0 - progress * 0.5)
        alpha = int(255 * (1.0 - progress))
        
        if alpha > 0 and current_size > 0:
            # 创建临时surface支持alpha
            temp_surface = pygame.Surface((int(current_size * 2), int(current_size * 2)), pygame.SRCALPHA)
            
            # 外圈（橙色）
            outer_color = (255, 165, 0, alpha)
            pygame.draw.circle(temp_surface, outer_color, 
                             (int(current_size), int(current_size)), int(current_size))
            
            # 内圈（黄色）
            inner_size = current_size * 0.6
            if inner_size > 0:
                inner_color = (255, 255, 0, alpha)
                pygame.draw.circle(temp_surface, inner_color, 
                                 (int(current_size), int(current_size)), int(inner_size))
            
            # 核心（白色）
            core_size = current_size * 0.3
            if core_size > 0:
                core_color = (255, 255, 255, alpha)
                pygame.draw.circle(temp_surface, core_color, 
                                 (int(current_size), int(current_size)), int(core_size))
            
            # 绘制到主surface
            pos = (int(self.position[0] - current_size + camera_offset[0]),
                   int(self.position[1] - current_size + camera_offset[1]))
            surface.blit(temp_surface, pos)
        
        # 渲染粒子
        for particle in self.particles:
            life_ratio = particle['life'] / particle['max_life']
            if life_ratio > 0:
                particle_size = int(particle['size'] * life_ratio)
                
                if particle_size > 0:
                    # 确保颜色值在有效范围内
                    red = 255
                    green = max(0, min(255, int(200 * life_ratio)))
                    blue = 0
                    color = (red, green, blue)
                    
                    pos = (int(particle['x'] + camera_offset[0]),
                           int(particle['y'] + camera_offset[1]))
                    
                    # 确保位置和大小都是有效的
                    if pos[0] >= 0 and pos[1] >= 0 and particle_size > 0:
                        try:
                            pygame.draw.circle(surface, color, pos, particle_size)
                        except ValueError:
                            # 如果还有问题，跳过这个粒子
                            continue


class TrailEffect(VisualEffect):
    """轨迹效果"""
    
    def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                 color: Tuple[int, int, int] = (100, 150, 255), width: int = 2, duration: float = 0.5):
        super().__init__(start_pos, duration)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.width = width
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        if not self.active:
            return
        
        progress = self.elapsed_time / self.duration
        alpha = int(255 * (1.0 - progress))
        
        if alpha > 0:
            start = (self.start_pos[0] + camera_offset[0], 
                    self.start_pos[1] + camera_offset[1])
            end = (self.end_pos[0] + camera_offset[0], 
                   self.end_pos[1] + camera_offset[1])
            
            # 使用渐变alpha的颜色
            color_with_alpha = (*self.color, alpha)
            
            # 创建临时surface来支持alpha
            temp_surface = pygame.Surface((abs(end[0] - start[0]) + self.width * 2,
                                         abs(end[1] - start[1]) + self.width * 2), 
                                        pygame.SRCALPHA)
            
            # 调整坐标到临时surface
            temp_start = (self.width, self.width) if start[0] <= end[0] else (temp_surface.get_width() - self.width, self.width)
            temp_end = (temp_surface.get_width() - self.width, temp_surface.get_height() - self.width) if start[0] <= end[0] else (self.width, temp_surface.get_height() - self.width)
            
            pygame.draw.line(temp_surface, color_with_alpha, temp_start, temp_end, self.width)
            
            # 绘制到主surface
            blit_pos = (min(start[0], end[0]) - self.width, min(start[1], end[1]) - self.width)
            surface.blit(temp_surface, blit_pos)


class MuzzleFlashEffect(VisualEffect):
    """炮口闪光效果"""
    
    def __init__(self, position: Tuple[float, float], direction: float, duration: float = 0.1):
        super().__init__(position, duration)
        self.direction = direction  # 角度（度）
        self.flash_length = 30
        self.flash_width = 15
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        if not self.active:
            return
        
        progress = self.elapsed_time / self.duration
        alpha = int(255 * (1.0 - progress))
        
        if alpha > 0:
            # 计算闪光的端点
            end_x = self.position[0] + math.cos(math.radians(self.direction)) * self.flash_length
            end_y = self.position[1] + math.sin(math.radians(self.direction)) * self.flash_length
            
            start = (self.position[0] + camera_offset[0], self.position[1] + camera_offset[1])
            end = (end_x + camera_offset[0], end_y + camera_offset[1])
            
            # 绘制闪光
            colors = [(255, 255, 255, alpha), (255, 255, 0, alpha), (255, 165, 0, alpha)]
            widths = [self.flash_width, self.flash_width // 2, self.flash_width // 4]
            
            for color, width in zip(colors, widths):
                if width > 0:
                    pygame.draw.line(surface, color[:3], start, end, width)


class VisualEffectsManager:
    """视觉效果管理器"""
    
    def __init__(self):
        self.effects: List[VisualEffect] = []
    
    def add_explosion(self, position: Tuple[float, float], size: float = 50.0):
        """添加爆炸效果"""
        effect = ExplosionEffect(position, size)
        self.effects.append(effect)
    
    def add_trail(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                  color: Tuple[int, int, int] = (100, 150, 255)):
        """添加轨迹效果"""
        effect = TrailEffect(start_pos, end_pos, color)
        self.effects.append(effect)
    
    def add_muzzle_flash(self, position: Tuple[float, float], direction: float):
        """添加炮口闪光效果"""
        effect = MuzzleFlashEffect(position, direction)
        self.effects.append(effect)
    
    def update(self, dt: float):
        """更新所有效果"""
        # 更新效果并移除已完成的
        self.effects = [effect for effect in self.effects if not effect.is_finished()]
        
        for effect in self.effects:
            effect.update(dt)
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """渲染所有效果"""
        for effect in self.effects:
            effect.render(surface, camera_offset)
    
    def clear_all(self):
        """清除所有效果"""
        self.effects.clear()
    
    def get_effect_count(self) -> int:
        """获取当前效果数量"""
        return len(self.effects)