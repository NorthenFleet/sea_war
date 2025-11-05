"""
分层渲染系统
支持Z轴排序、多层级显示和高效的渲染管理
使用pygame.sprite.LayeredUpdates实现分层渲染
"""

import pygame
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import IntEnum
from collections import defaultdict
from ..core.entities.entity import *
from .entity_sprite import EntitySprite


class RenderLayer(IntEnum):
    """渲染层级定义（数值越大越靠前）"""
    BACKGROUND = 0          # 背景层（地形、海洋）
    TERRAIN_DETAILS = 10    # 地形细节（岛屿、礁石）
    UNDERWATER = 20         # 水下单位（潜艇）
    SURFACE = 30           # 水面单位（舰船）
    AIRBORNE_LOW = 40      # 低空单位（导弹、鱼雷）
    AIRBORNE_HIGH = 50     # 高空单位（飞机）
    EFFECTS_LOW = 60       # 低层特效（爆炸、水花）
    EFFECTS_HIGH = 70      # 高层特效（烟雾、火焰）
    UI_BACKGROUND = 80     # UI背景层
    UI_ELEMENTS = 90       # UI元素
    UI_OVERLAY = 100       # UI覆盖层（选择框、提示）
    DEBUG = 110            # 调试信息


class LayeredRenderManager:
    """
    分层渲染管理器
    管理多个渲染层，支持高效的Z轴排序和批量渲染
    """
    
    def __init__(self, screen: pygame.Surface, sprite_loader=None, camera_offset: Tuple[int, int] = (0, 0)):
        self.screen = screen
        self.camera_offset = list(camera_offset)
        
        # 精灵工厂
        from .entity_sprite import SpriteFactory
        self.sprite_factory = SpriteFactory(sprite_loader) if sprite_loader else None
        
        # 分层精灵组
        self.layered_group = pygame.sprite.LayeredUpdates()
        
        # 按层分组的精灵组（用于快速访问特定层）
        self.layer_groups: Dict[RenderLayer, pygame.sprite.Group] = {
            layer: pygame.sprite.Group() for layer in RenderLayer
        }
        
        # 实体到精灵的映射
        self.entity_sprites: Dict[int, EntitySprite] = {}
        
        # 渲染表面缓存
        self.layer_surfaces: Dict[RenderLayer, pygame.Surface] = {}
        self.surface_cache_enabled = True
        self.cache_dirty_flags: Dict[RenderLayer, bool] = {
            layer: True for layer in RenderLayer
        }
        
        # 渲染区域管理
        self.viewport = pygame.Rect(0, 0, screen.get_width(), screen.get_height())
        self.culling_enabled = True
        self.culling_margin = 100  # 视口外边距，用于预加载
        
        # 性能统计
        self.render_stats = {
            'total_sprites': 0,
            'visible_sprites': 0,
            'culled_sprites': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'layer_renders': defaultdict(int)
        }
        
        # 特效管理
        self.effects: List['RenderEffect'] = []
        
        # 初始化层表面
        self._initialize_layer_surfaces()
    
    def _initialize_layer_surfaces(self):
        """初始化层渲染表面"""
        screen_size = self.screen.get_size()
        
        for layer in RenderLayer:
            # 根据层类型决定是否需要透明度
            if layer in [RenderLayer.EFFECTS_LOW, RenderLayer.EFFECTS_HIGH, 
                        RenderLayer.UI_OVERLAY, RenderLayer.DEBUG]:
                surface = pygame.Surface(screen_size, pygame.SRCALPHA)
            else:
                surface = pygame.Surface(screen_size)
            
            surface = surface.convert_alpha()
            self.layer_surfaces[layer] = surface
    
    def add_entity(self, entity, layer: RenderLayer = None) -> Optional[EntitySprite]:
        """
        添加实体到渲染系统
        
        Args:
            entity: ECS实体
            layer: 渲染层级，None表示自动判断
        
        Returns:
            创建的EntitySprite实例
        """
        if entity.entity_id in self.entity_sprites:
            return self.entity_sprites[entity.entity_id]
        
        # 自动判断层级
        if layer is None:
            layer = self._determine_entity_layer(entity)
        
        # 创建精灵
        if self.sprite_factory:
            sprite = self.sprite_factory.create_sprite(entity)
        else:
            # 如果没有sprite_factory，创建默认精灵
            sprite = EntitySprite(entity, None)
        
        # 添加到分层组
        self.layered_group.add(sprite, layer=layer.value)
        self.layer_groups[layer].add(sprite)
        
        # 保存映射
        self.entity_sprites[entity.entity_id] = sprite
        
        # 标记层为脏
        self.cache_dirty_flags[layer] = True
        
        return sprite
    
    def remove_entity(self, entity_id: int):
        """从渲染系统移除实体"""
        if entity_id not in self.entity_sprites:
            return
        
        sprite = self.entity_sprites[entity_id]
        layer = RenderLayer(self.layered_group.get_layer_of_sprite(sprite))
        
        # 从所有组中移除
        self.layered_group.remove(sprite)
        self.layer_groups[layer].remove(sprite)
        
        # 清理映射
        del self.entity_sprites[entity_id]
        
        # 标记层为脏
        self.cache_dirty_flags[layer] = True
    
    def _determine_entity_layer(self, entity) -> RenderLayer:
        """自动判断实体的渲染层级"""
        entity_type = getattr(entity, 'entity_type', 'unknown')
        
        # 根据实体类型判断层级
        layer_mapping = {
            'submarine': RenderLayer.UNDERWATER,
            'ship': RenderLayer.SURFACE,
            'destroyer': RenderLayer.SURFACE,
            'cruiser': RenderLayer.SURFACE,
            'carrier': RenderLayer.SURFACE,
            'missile': RenderLayer.AIRBORNE_LOW,
            'torpedo': RenderLayer.UNDERWATER,
            'bomber': RenderLayer.AIRBORNE_HIGH,
            'fighter': RenderLayer.AIRBORNE_HIGH,
            'ground_based_platforms': RenderLayer.TERRAIN_DETAILS,
            'airport': RenderLayer.TERRAIN_DETAILS
        }
        
        return layer_mapping.get(entity_type, RenderLayer.SURFACE)
    
    def update_camera(self, offset: Tuple[int, int]):
        """更新摄像机偏移"""
        if tuple(self.camera_offset) != offset:
            self.camera_offset = list(offset)
            # 摄像机移动时，所有层都需要重新渲染
            for layer in RenderLayer:
                self.cache_dirty_flags[layer] = True
    
    def update_viewport(self, viewport: pygame.Rect):
        """更新视口"""
        if self.viewport != viewport:
            self.viewport = viewport.copy()
            # 视口变化时，需要重新计算可见性
            for layer in RenderLayer:
                self.cache_dirty_flags[layer] = True
    
    def add_effect(self, effect: 'RenderEffect'):
        """添加渲染特效"""
        self.effects.append(effect)
        # 标记特效层为脏
        self.cache_dirty_flags[effect.layer] = True
    
    def update(self, delta_time: float):
        """更新渲染系统"""
        # 重置统计
        self.render_stats['total_sprites'] = len(self.entity_sprites)
        self.render_stats['visible_sprites'] = 0
        self.render_stats['culled_sprites'] = 0
        
        # 更新所有精灵
        for sprite in self.entity_sprites.values():
            sprite.update(delta_time)
            
            # 检查是否在视口内
            if self.culling_enabled:
                sprite_rect = sprite.rect.copy()
                sprite_rect.x += self.camera_offset[0]
                sprite_rect.y += self.camera_offset[1]
                
                expanded_viewport = self.viewport.inflate(
                    self.culling_margin * 2, self.culling_margin * 2
                )
                
                if expanded_viewport.colliderect(sprite_rect):
                    self.render_stats['visible_sprites'] += 1
                    sprite.visible = True
                else:
                    self.render_stats['culled_sprites'] += 1
                    sprite.visible = False
            else:
                sprite.visible = True
                self.render_stats['visible_sprites'] += 1
        
        # 更新特效
        self.effects = [effect for effect in self.effects if effect.update(delta_time)]
    
    def render(self, target_surface: pygame.Surface = None, dt: float = 0.0):
        """执行分层渲染"""
        if target_surface is None:
            target_surface = self.screen
        
        # 如果提供了delta_time，先更新特效
        if dt > 0.0:
            self.update(dt)
        
        # 按层级顺序渲染
        for layer in sorted(RenderLayer, key=lambda x: x.value):
            self._render_layer(layer, target_surface)
    
    def _render_layer(self, layer: RenderLayer, target_surface: pygame.Surface):
        """渲染指定层"""
        layer_surface = self.layer_surfaces[layer]
        
        # 检查是否需要重新渲染层
        if self.surface_cache_enabled and not self.cache_dirty_flags[layer]:
            # 使用缓存
            target_surface.blit(layer_surface, (0, 0))
            self.render_stats['cache_hits'] += 1
            return
        
        # 重新渲染层
        self.render_stats['cache_misses'] += 1
        self.render_stats['layer_renders'][layer] += 1
        
        # 清空层表面
        if layer in [RenderLayer.EFFECTS_LOW, RenderLayer.EFFECTS_HIGH, 
                    RenderLayer.UI_OVERLAY, RenderLayer.DEBUG]:
            layer_surface.fill((0, 0, 0, 0))  # 透明
        else:
            layer_surface.fill((0, 0, 0))  # 黑色
        
        # 渲染该层的精灵
        layer_group = self.layer_groups[layer]
        visible_sprites = [sprite for sprite in layer_group if getattr(sprite, 'visible', True)]
        
        for sprite in visible_sprites:
            # 应用摄像机偏移
            render_pos = (
                sprite.rect.x + self.camera_offset[0],
                sprite.rect.y + self.camera_offset[1]
            )
            layer_surface.blit(sprite.image, render_pos)
        
        # 渲染该层的特效
        layer_effects = [effect for effect in self.effects if effect.layer == layer]
        for effect in layer_effects:
            effect.render(layer_surface, self.camera_offset)
        
        # 将层表面绘制到目标表面
        target_surface.blit(layer_surface, (0, 0))
        
        # 标记层为干净
        self.cache_dirty_flags[layer] = False
    
    def render_layer_direct(self, layer: RenderLayer, target_surface: pygame.Surface):
        """直接渲染指定层（不使用缓存）"""
        layer_group = self.layer_groups[layer]
        visible_sprites = [sprite for sprite in layer_group if getattr(sprite, 'visible', True)]
        
        for sprite in visible_sprites:
            render_pos = (
                sprite.rect.x + self.camera_offset[0],
                sprite.rect.y + self.camera_offset[1]
            )
            target_surface.blit(sprite.image, render_pos)
        
        # 渲染特效
        layer_effects = [effect for effect in self.effects if effect.layer == layer]
        for effect in layer_effects:
            effect.render(target_surface, self.camera_offset)
    
    def get_sprites_in_layer(self, layer: RenderLayer) -> List[EntitySprite]:
        """获取指定层的所有精灵"""
        return list(self.layer_groups[layer])
    
    def get_sprites_at_position(self, pos: Tuple[int, int], layer: RenderLayer = None) -> List[EntitySprite]:
        """获取指定位置的精灵"""
        # 转换为世界坐标
        world_pos = (pos[0] - self.camera_offset[0], pos[1] - self.camera_offset[1])
        
        results = []
        
        if layer is not None:
            # 只检查指定层
            sprites_to_check = self.layer_groups[layer]
        else:
            # 检查所有层，按Z轴顺序（从前到后）
            sprites_to_check = []
            for layer_enum in sorted(RenderLayer, key=lambda x: x.value, reverse=True):
                sprites_to_check.extend(self.layer_groups[layer_enum])
        
        for sprite in sprites_to_check:
            if sprite.rect.collidepoint(world_pos) and getattr(sprite, 'visible', True):
                results.append(sprite)
        
        return results
    
    def set_layer_visibility(self, layer: RenderLayer, visible: bool):
        """设置层的可见性"""
        for sprite in self.layer_groups[layer]:
            sprite.visible = visible
        
        # 标记层为脏
        self.cache_dirty_flags[layer] = True
    
    def clear_layer(self, layer: RenderLayer):
        """清空指定层"""
        sprites_to_remove = list(self.layer_groups[layer])
        for sprite in sprites_to_remove:
            entity_id = sprite.entity.entity_id
            self.remove_entity(entity_id)
    
    def get_render_stats(self) -> Dict[str, Any]:
        """获取渲染统计信息"""
        return {
            'performance': self.render_stats.copy(),
            'layer_info': {
                layer.name: len(self.layer_groups[layer]) 
                for layer in RenderLayer
            },
            'cache_enabled': self.surface_cache_enabled,
            'culling_enabled': self.culling_enabled,
            'effects_count': len(self.effects)
        }
    
    def toggle_surface_cache(self):
        """切换表面缓存"""
        self.surface_cache_enabled = not self.surface_cache_enabled
        if self.surface_cache_enabled:
            # 启用缓存时，标记所有层为脏
            for layer in RenderLayer:
                self.cache_dirty_flags[layer] = True
    
    def toggle_culling(self):
        """切换视锥剔除"""
        self.culling_enabled = not self.culling_enabled


class RenderEffect:
    """
    渲染特效基类
    用于实现各种视觉特效
    """
    
    def __init__(self, position: Tuple[float, float], layer: RenderLayer = RenderLayer.EFFECTS_LOW, 
                 duration: float = 1.0):
        self.position = position
        self.layer = layer
        self.duration = duration
        self.elapsed_time = 0.0
        self.active = True
    
    def update(self, delta_time: float) -> bool:
        """
        更新特效
        
        Returns:
            True表示特效仍然活跃，False表示应该移除
        """
        if not self.active:
            return False
        
        self.elapsed_time += delta_time
        
        if self.elapsed_time >= self.duration:
            self.active = False
            return False
        
        return True
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int]):
        """渲染特效"""
        pass
    
    def get_progress(self) -> float:
        """获取特效进度（0.0到1.0）"""
        if self.duration <= 0:
            return 1.0
        return min(1.0, self.elapsed_time / self.duration)


class ExplosionEffect(RenderEffect):
    """爆炸特效"""
    
    def __init__(self, position: Tuple[float, float], size: float = 50.0, 
                 color: Tuple[int, int, int] = (255, 100, 0), duration: float = 0.5):
        super().__init__(position, RenderLayer.EFFECTS_HIGH, duration)
        self.size = size
        self.color = color
        self.max_radius = size
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int]):
        if not self.active:
            return
        
        progress = self.get_progress()
        
        # 计算当前半径和透明度
        current_radius = int(self.max_radius * progress)
        alpha = int(255 * (1.0 - progress))
        
        # 创建带透明度的表面
        explosion_surface = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
        
        # 绘制爆炸圆圈
        color_with_alpha = (*self.color, alpha)
        pygame.draw.circle(explosion_surface, color_with_alpha, 
                         (current_radius, current_radius), current_radius)
        
        # 绘制到目标表面
        render_pos = (
            self.position[0] + camera_offset[0] - current_radius,
            self.position[1] + camera_offset[1] - current_radius
        )
        surface.blit(explosion_surface, render_pos)


class TrailEffect(RenderEffect):
    """轨迹特效"""
    
    def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float],
                 width: int = 3, color: Tuple[int, int, int] = (255, 255, 255), 
                 duration: float = 1.0):
        super().__init__(start_pos, RenderLayer.EFFECTS_LOW, duration)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.width = width
        self.color = color
    
    def render(self, surface: pygame.Surface, camera_offset: Tuple[int, int]):
        if not self.active:
            return
        
        progress = self.get_progress()
        alpha = int(255 * (1.0 - progress))
        
        # 计算当前轨迹终点
        current_end = (
            self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress,
            self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        )
        
        # 应用摄像机偏移
        screen_start = (
            self.start_pos[0] + camera_offset[0],
            self.start_pos[1] + camera_offset[1]
        )
        screen_end = (
            current_end[0] + camera_offset[0],
            current_end[1] + camera_offset[1]
        )
        
        # 绘制轨迹线
        if alpha > 0:
            color_with_alpha = (*self.color, alpha)
            # 注意：pygame.draw.line不支持alpha，需要使用表面混合
            trail_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.line(trail_surface, color_with_alpha, screen_start, screen_end, self.width)
            surface.blit(trail_surface, (0, 0))


class LayeredRenderDebugger:
    """分层渲染调试器"""
    
    def __init__(self, render_manager: LayeredRenderManager):
        self.render_manager = render_manager
        self.show_layer_info = False
        self.show_performance = False
        self.show_bounds = False
        self.font = pygame.font.Font(None, 20)
    
    def toggle_layer_info(self):
        """切换层信息显示"""
        self.show_layer_info = not self.show_layer_info
    
    def toggle_performance(self):
        """切换性能信息显示"""
        self.show_performance = not self.show_performance
    
    def toggle_bounds(self):
        """切换边界显示"""
        self.show_bounds = not self.show_bounds
    
    def render_debug_info(self, surface: pygame.Surface):
        """渲染调试信息"""
        if self.show_layer_info:
            self._render_layer_info(surface)
        
        if self.show_performance:
            self._render_performance_info(surface)
        
        if self.show_bounds:
            self._render_sprite_bounds(surface)
    
    def _render_layer_info(self, surface: pygame.Surface):
        """渲染层信息"""
        y_offset = 10
        
        for layer in RenderLayer:
            count = len(self.render_manager.layer_groups[layer])
            if count > 0:
                text = f"{layer.name}: {count} sprites"
                color = (255, 255, 255) if count > 0 else (128, 128, 128)
                rendered_text = self.font.render(text, True, color)
                surface.blit(rendered_text, (10, y_offset))
                y_offset += 22
    
    def _render_performance_info(self, surface: pygame.Surface):
        """渲染性能信息"""
        stats = self.render_manager.get_render_stats()
        
        info_lines = [
            f"Total Sprites: {stats['performance']['total_sprites']}",
            f"Visible: {stats['performance']['visible_sprites']}",
            f"Culled: {stats['performance']['culled_sprites']}",
            f"Cache Hits: {stats['performance']['cache_hits']}",
            f"Cache Misses: {stats['performance']['cache_misses']}",
            f"Effects: {stats['effects_count']}"
        ]
        
        x_offset = surface.get_width() - 200
        y_offset = 10
        
        for line in info_lines:
            rendered_text = self.font.render(line, True, (255, 255, 0))
            surface.blit(rendered_text, (x_offset, y_offset))
            y_offset += 22
    
    def _render_sprite_bounds(self, surface: pygame.Surface):
        """渲染精灵边界"""
        for sprite in self.render_manager.entity_sprites.values():
            if getattr(sprite, 'visible', True):
                rect = sprite.rect.copy()
                rect.x += self.render_manager.camera_offset[0]
                rect.y += self.render_manager.camera_offset[1]
                pygame.draw.rect(surface, (0, 255, 0), rect, 1)