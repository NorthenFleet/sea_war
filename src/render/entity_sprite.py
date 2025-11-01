"""
Entity到pygame.sprite.Sprite的适配器系统
提供ECS实体与pygame精灵系统之间的桥梁
"""

import pygame
import math
from ..core.entities.entity import PositionComponent, HealthComponent, MovementComponent


class EntitySprite(pygame.sprite.Sprite):
    """
    将ECS实体适配为pygame精灵的适配器类
    自动同步实体的位置、状态和外观
    """
    
    def __init__(self, entity, image, scale_factor=1.0):
        super().__init__()
        self.entity = entity
        self.entity_id = entity.entity_id
        self.entity_type = entity.entity_type
        
        # 图像处理和优化
        if image:
            # 使用convert_alpha()优化渲染性能
            self.original_image = image.convert_alpha()
            if scale_factor != 1.0:
                new_size = (int(image.get_width() * scale_factor), 
                           int(image.get_height() * scale_factor))
                self.original_image = pygame.transform.smoothscale(self.original_image, new_size)
        else:
            # 创建默认占位图像
            self.original_image = self._create_default_image()
        
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect()
        
        # 缓存组件引用以提高性能
        self._position_component = None
        self._health_component = None
        self._movement_component = None
        self._update_component_cache()
        
        # 渲染状态
        self.selected = False
        self.last_position = None
        self.last_heading = None
        self.rotation_angle = 0
        
        # 视觉增强
        self.show_direction_indicator = True
        self.show_status_effects = True
        self.trail_points = []  # 轨迹点
        self.max_trail_length = 10
        
        # 状态效果
        self.status_effects = {
            'damaged': False,
            'moving': False,
            'attacking': False,
            'low_health': False
        }
        
        # 初始化位置
        self._update_position()
    
    def _create_default_image(self):
        """为未知类型创建默认图像"""
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
        
        size = size_map.get(self.entity_type, (28, 28))
        color = color_map.get(self.entity_type, (200, 200, 200))
        
        surface = pygame.Surface(size, pygame.SRCALPHA)
        surface.fill(color)
        
        # 添加简单的形状标识
        if self.entity_type in ('ship', 'submarine'):
            pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 1)
        elif self.entity_type == 'missile':
            pygame.draw.polygon(surface, (255, 255, 255), 
                              [(size[0]//2, 0), (size[0], size[1]//2), (0, size[1]//2)])
        
        return surface.convert_alpha()
    
    def _update_component_cache(self):
        """更新组件缓存以提高性能"""
        self._position_component = self.entity.get_component(PositionComponent)
        self._health_component = self.entity.get_component(HealthComponent)
        self._movement_component = self.entity.get_component(MovementComponent)
    
    def _update_position(self):
        """更新精灵位置"""
        if self._position_component:
            pos = self._position_component.get_param('position')
            if pos and len(pos) >= 2:
                # 只有位置真正改变时才更新
                new_pos = (int(pos[0]), int(pos[1]))
                if self.last_position != new_pos:
                    self.rect.center = new_pos
                    self.last_position = new_pos
    
    def _update_rotation(self):
        """根据移动方向更新精灵旋转"""
        if self._movement_component:
            heading = self._movement_component.get_param('heading')
            if heading and len(heading) >= 2:
                # 计算旋转角度
                angle = math.degrees(math.atan2(heading[1], heading[0]))
                if abs(angle - self.last_heading) > 1:  # 只有角度变化超过1度才更新
                    self.last_heading = angle
                    self.rotation_angle = angle
                    
                    # 旋转图像
                    rotated_image = pygame.transform.rotate(self.original_image, -angle)
                    old_center = self.rect.center
                    self.image = rotated_image
                    self.rect = self.image.get_rect()
                    self.rect.center = old_center
    
    def _update_trail(self):
        """更新轨迹点"""
        if self._position_component:
            current_pos = (
                self._position_component.get_param('x', 0),
                self._position_component.get_param('y', 0)
            )
            
            # 只有位置发生变化时才添加轨迹点
            if self.last_position and current_pos != self.last_position:
                self.trail_points.append(current_pos)
                
                # 限制轨迹长度
                if len(self.trail_points) > self.max_trail_length:
                    self.trail_points.pop(0)
    
    def _update_status_effects(self):
        """更新状态效果"""
        # 检查生命值状态
        if self._health_component:
            max_hp = self._health_component.get_param('max_health', 100)
            cur_hp = self._health_component.get_param('current_health', 100)
            
            if max_hp > 0:
                health_ratio = cur_hp / max_hp
                self.status_effects['low_health'] = health_ratio < 0.3
                self.status_effects['damaged'] = health_ratio < 0.8
        
        # 检查移动状态
        if self._movement_component:
            speed = self._movement_component.get_param('speed', 0)
            self.status_effects['moving'] = speed > 0.1
        
        # 检查攻击状态（可以根据实际游戏逻辑扩展）
        # 这里简化处理，可以根据实体的其他组件来判断
        self.status_effects['attacking'] = False
    
    def update(self, *args, **kwargs):
        """pygame精灵的标准更新方法"""
        # 检查实体是否仍然存在
        if not hasattr(self.entity, 'entity_id'):
            self.kill()
            return
        
        # 更新组件缓存
        self._update_component_cache()
        
        # 记录轨迹点
        self._update_trail()
        
        # 更新状态效果
        self._update_status_effects()
        
        # 更新位置
        self._update_position()
        
        # 更新旋转（如果需要）
        if self.entity_type in ('ship', 'submarine', 'missile', 'bomber'):
            self._update_rotation()
    
    def set_selected(self, selected):
        """设置选中状态"""
        self.selected = selected
    
    def draw_health_bar(self, surface, offset=(0, 0)):
        """绘制血量条"""
        if not self._health_component:
            return
        
        max_hp = self._health_component.get_param('max_health')
        cur_hp = self._health_component.get_param('current_health')
        
        if not max_hp or max_hp <= 0:
            return
        
        ratio = max(0.0, min(1.0, cur_hp / max_hp))
        
        # 血量条尺寸和位置
        bar_width = self.rect.width
        bar_height = 4
        bar_x = self.rect.x + offset[0]
        bar_y = self.rect.y + offset[1] - 8
        
        # 背景
        pygame.draw.rect(surface, (60, 60, 60), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # 前景
        fg_width = int(bar_width * ratio)
        if ratio > 0.6:
            color = (40, 200, 40)  # 绿色
        elif ratio > 0.3:
            color = (200, 180, 40)  # 黄色
        else:
            color = (200, 40, 40)  # 红色
        
        pygame.draw.rect(surface, color, 
                        (bar_x, bar_y, fg_width, bar_height))
    
    def draw_selection_indicator(self, surface, offset=(0, 0)):
        """绘制选中指示器"""
        if self.selected:
            rect = self.rect.copy()
            rect.x += offset[0]
            rect.y += offset[1]
            pygame.draw.rect(surface, (255, 255, 0), rect, 2)
    
    def draw_direction_indicator(self, surface, offset=(0, 0)):
        """绘制方向指示器"""
        if not self.show_direction_indicator or not self._movement_component:
            return
        
        heading = self._movement_component.get_param('heading', 0)
        speed = self._movement_component.get_param('speed', 0)
        
        if speed > 0.1:  # 只有在移动时才显示方向
            center_x = self.rect.centerx + offset[0]
            center_y = self.rect.centery + offset[1]
            
            # 计算箭头终点
            arrow_length = 30
            end_x = center_x + math.cos(math.radians(heading)) * arrow_length
            end_y = center_y + math.sin(math.radians(heading)) * arrow_length
            
            # 绘制方向箭头
            pygame.draw.line(surface, (0, 255, 255), 
                           (center_x, center_y), (end_x, end_y), 2)
            
            # 绘制箭头头部
            arrow_head_length = 8
            arrow_angle = 30
            
            # 左侧箭头
            left_angle = heading + 180 - arrow_angle
            left_x = end_x + math.cos(math.radians(left_angle)) * arrow_head_length
            left_y = end_y + math.sin(math.radians(left_angle)) * arrow_head_length
            pygame.draw.line(surface, (0, 255, 255), (end_x, end_y), (left_x, left_y), 2)
            
            # 右侧箭头
            right_angle = heading + 180 + arrow_angle
            right_x = end_x + math.cos(math.radians(right_angle)) * arrow_head_length
            right_y = end_y + math.sin(math.radians(right_angle)) * arrow_head_length
            pygame.draw.line(surface, (0, 255, 255), (end_x, end_y), (right_x, right_y), 2)
    
    def draw_trail(self, surface, offset=(0, 0)):
        """绘制移动轨迹"""
        if len(self.trail_points) < 2:
            return
        
        # 绘制轨迹线
        for i in range(1, len(self.trail_points)):
            start_pos = (
                self.trail_points[i-1][0] + offset[0],
                self.trail_points[i-1][1] + offset[1]
            )
            end_pos = (
                self.trail_points[i][0] + offset[0],
                self.trail_points[i][1] + offset[1]
            )
            
            # 轨迹透明度随时间衰减
            alpha = int(255 * (i / len(self.trail_points)) * 0.5)
            color = (100, 150, 255, alpha)
            
            # 创建临时surface来支持alpha
            temp_surface = pygame.Surface((2, 2), pygame.SRCALPHA)
            temp_surface.fill(color)
            
            pygame.draw.line(surface, color[:3], start_pos, end_pos, 1)
    
    def draw_status_effects(self, surface, offset=(0, 0)):
        """绘制状态效果"""
        if not self.show_status_effects:
            return
        
        effect_y = self.rect.y + offset[1] - 20
        effect_x = self.rect.x + offset[0]
        
        # 低血量警告
        if self.status_effects['low_health']:
            pygame.draw.circle(surface, (255, 0, 0), 
                             (effect_x + 5, effect_y), 3)
            effect_x += 12
        
        # 受损状态
        if self.status_effects['damaged'] and not self.status_effects['low_health']:
            pygame.draw.circle(surface, (255, 165, 0), 
                             (effect_x + 5, effect_y), 3)
            effect_x += 12
        
        # 移动状态
        if self.status_effects['moving']:
            pygame.draw.polygon(surface, (0, 255, 0), [
                (effect_x + 2, effect_y + 3),
                (effect_x + 8, effect_y),
                (effect_x + 8, effect_y + 6)
            ])
            effect_x += 12
        
        # 攻击状态
        if self.status_effects['attacking']:
            pygame.draw.circle(surface, (255, 255, 0), 
                             (effect_x + 5, effect_y), 3)
    
    def draw_enhanced_effects(self, surface, offset=(0, 0)):
        """绘制所有增强视觉效果"""
        self.draw_trail(surface, offset)
        self.draw_direction_indicator(surface, offset)
        self.draw_status_effects(surface, offset)
        self.draw_health_bar(surface, offset)
        self.draw_selection_indicator(surface, offset)
    
    def get_collision_rect(self):
        """获取碰撞检测矩形"""
        return self.rect
    
    def __repr__(self):
        return f"EntitySprite(id={self.entity_id}, type={self.entity_type}, pos={self.rect.center})"


class SpriteFactory:
    """
    精灵工厂类，负责创建和管理EntitySprite实例
    """
    
    def __init__(self, sprite_loader):
        self.sprite_loader = sprite_loader
        self.sprite_cache = {}
    
    def create_sprite(self, entity, scale_factor=1.0):
        """为实体创建对应的精灵"""
        # 获取实体类型对应的图像
        image = self.sprite_loader.get(entity.entity_type)
        
        # 创建精灵
        sprite = EntitySprite(entity, image, scale_factor)
        
        # 缓存精灵
        self.sprite_cache[entity.entity_id] = sprite
        
        return sprite
    
    def get_sprite(self, entity_id):
        """获取缓存的精灵"""
        return self.sprite_cache.get(entity_id)
    
    def remove_sprite(self, entity_id):
        """移除精灵"""
        if entity_id in self.sprite_cache:
            sprite = self.sprite_cache[entity_id]
            sprite.kill()  # 从所有精灵组中移除
            del self.sprite_cache[entity_id]
    
    def clear_cache(self):
        """清空精灵缓存"""
        for sprite in self.sprite_cache.values():
            sprite.kill()
        self.sprite_cache.clear()