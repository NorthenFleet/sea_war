"""
pygame精灵组管理系统
提供按类型和层级组织的精灵管理，支持高效的批量渲染和碰撞检测
"""

import pygame
from .entity_sprite import EntitySprite, SpriteFactory
from collections import defaultdict


class LayeredSpriteManager:
    """
    分层精灵管理器
    使用pygame.sprite.LayeredUpdates实现Z轴排序和分层渲染
    """
    
    # 定义渲染层级
    LAYER_BACKGROUND = 0
    LAYER_TERRAIN = 1
    LAYER_SHIPS = 2
    LAYER_SUBMARINES = 3
    LAYER_AIRCRAFT = 4
    LAYER_MISSILES = 5
    LAYER_EFFECTS = 6
    LAYER_UI = 7
    
    # 实体类型到层级的映射
    ENTITY_LAYER_MAP = {
        'ship': LAYER_SHIPS,
        'submarine': LAYER_SUBMARINES,
        'bomber': LAYER_AIRCRAFT,
        'missile': LAYER_MISSILES,
        'ground_based_platforms': LAYER_TERRAIN,
        'airport': LAYER_TERRAIN,
    }
    
    def __init__(self, sprite_loader):
        # 主要的分层精灵组
        self.all_sprites = pygame.sprite.LayeredUpdates()
        
        # 按类型分组的精灵组，用于特定操作
        self.sprite_groups = {
            'ships': pygame.sprite.Group(),
            'submarines': pygame.sprite.Group(),
            'aircraft': pygame.sprite.Group(),
            'missiles': pygame.sprite.Group(),
            'platforms': pygame.sprite.Group(),
            'all_units': pygame.sprite.Group(),  # 所有可移动单位
            'all_static': pygame.sprite.Group(),  # 所有静态单位
        }
        
        # 精灵工厂
        self.sprite_factory = SpriteFactory(sprite_loader)
        
        # 实体ID到精灵的映射
        self.entity_sprite_map = {}
        
        # 选中的精灵
        self.selected_sprites = pygame.sprite.Group()
        
        # 碰撞检测组
        self.collision_groups = {
            'friendly': pygame.sprite.Group(),
            'enemy': pygame.sprite.Group(),
            'neutral': pygame.sprite.Group(),
        }
        
        # 性能优化参数
        self.last_update_time = 0
        self.update_interval = 1.0 / 60.0  # 60 FPS更新频率
        self.dirty_sprites = set()  # 需要更新的精灵
    
    def add_entity(self, entity, scale_factor=1.0):
        """添加实体精灵"""
        if entity.entity_id in self.entity_sprite_map:
            return  # 已存在
        
        # 创建精灵
        sprite = self.sprite_factory.create_sprite(entity, scale_factor)
        
        # 确定层级
        layer = self.ENTITY_LAYER_MAP.get(entity.entity_type, self.LAYER_TERRAIN)
        
        # 添加到主精灵组
        self.all_sprites.add(sprite, layer=layer)
        
        # 添加到类型分组
        self._add_to_type_groups(sprite, entity.entity_type)
        
        # 添加到碰撞组（基于阵营）
        self._add_to_collision_groups(sprite, entity)
        
        # 记录映射
        self.entity_sprite_map[entity.entity_id] = sprite
        
        return sprite
    
    def _add_to_type_groups(self, sprite, entity_type):
        """将精灵添加到类型分组"""
        if entity_type == 'ship':
            self.sprite_groups['ships'].add(sprite)
            self.sprite_groups['all_units'].add(sprite)
        elif entity_type == 'submarine':
            self.sprite_groups['submarines'].add(sprite)
            self.sprite_groups['all_units'].add(sprite)
        elif entity_type in ('bomber',):
            self.sprite_groups['aircraft'].add(sprite)
            self.sprite_groups['all_units'].add(sprite)
        elif entity_type == 'missile':
            self.sprite_groups['missiles'].add(sprite)
            self.sprite_groups['all_units'].add(sprite)
        elif entity_type in ('ground_based_platforms', 'airport'):
            self.sprite_groups['platforms'].add(sprite)
            self.sprite_groups['all_static'].add(sprite)
    
    def _add_to_collision_groups(self, sprite, entity):
        """将精灵添加到碰撞组（基于阵营）"""
        # 这里可以根据实体的阵营属性来分组
        # 暂时使用简单的逻辑
        if hasattr(entity, 'team_id'):
            if entity.team_id == 1:
                self.collision_groups['friendly'].add(sprite)
            elif entity.team_id == 2:
                self.collision_groups['enemy'].add(sprite)
            else:
                self.collision_groups['neutral'].add(sprite)
        else:
            self.collision_groups['neutral'].add(sprite)
    
    def remove_entity(self, entity_id):
        """移除实体精灵"""
        if entity_id not in self.entity_sprite_map:
            return
        
        sprite = self.entity_sprite_map[entity_id]
        
        # 从所有组中移除
        sprite.kill()  # 这会从所有包含该精灵的组中移除
        
        # 从映射中移除
        del self.entity_sprite_map[entity_id]
        
        # 从工厂缓存中移除
        self.sprite_factory.remove_sprite(entity_id)
    
    def update_entities(self, entities):
        """批量更新实体精灵（优化版本）"""
        import time
        current_time = time.time()
        
        # 频率控制：只在达到更新间隔时才进行更新
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 获取当前存在的实体ID
        current_entity_ids = {entity.entity_id for entity in entities}
        existing_entity_ids = set(self.entity_sprite_map.keys())
        
        # 计算需要移除和添加的实体
        to_remove = existing_entity_ids - current_entity_ids
        to_add = current_entity_ids - existing_entity_ids
        
        # 批量移除不再存在的实体精灵
        for entity_id in to_remove:
            self.remove_entity(entity_id)
        
        # 批量添加新的实体精灵
        entities_to_add = {entity.entity_id: entity for entity in entities}
        for entity_id in to_add:
            if entity_id in entities_to_add:
                self.add_entity(entities_to_add[entity_id])
                self.dirty_sprites.add(entity_id)
        
        # 只更新有变化的精灵
        if self.dirty_sprites:
            # 只更新脏精灵而不是所有精灵
            for entity_id in list(self.dirty_sprites):
                if entity_id in self.entity_sprite_map:
                    sprite = self.entity_sprite_map[entity_id]
                    sprite.update()
            self.dirty_sprites.clear()
        elif to_remove or to_add:
            # 如果有添加或移除操作，更新所有精灵
            self.all_sprites.update()
    
    def render(self, surface, camera_offset=(0, 0)):
        """渲染所有精灵"""
        # 如果有摄像机偏移，需要调整精灵位置
        if camera_offset != (0, 0):
            self._apply_camera_offset(camera_offset)
        
        # 使用LayeredUpdates的draw方法进行分层渲染
        self.all_sprites.draw(surface)
        
        # 绘制额外的UI元素（血量条、选中指示器等）
        self._render_ui_elements(surface, camera_offset)
        
        # 恢复精灵位置
        if camera_offset != (0, 0):
            self._restore_camera_offset(camera_offset)
    
    def _apply_camera_offset(self, offset):
        """应用摄像机偏移"""
        for sprite in self.all_sprites:
            sprite.rect.x += offset[0]
            sprite.rect.y += offset[1]
    
    def _restore_camera_offset(self, offset):
        """恢复摄像机偏移"""
        for sprite in self.all_sprites:
            sprite.rect.x -= offset[0]
            sprite.rect.y -= offset[1]
    
    def _render_ui_elements(self, surface, camera_offset):
        """渲染UI元素（血量条、选中指示器等）"""
        for sprite in self.all_sprites:
            # 绘制血量条
            sprite.draw_health_bar(surface, camera_offset)
            
            # 绘制选中指示器
            sprite.draw_selection_indicator(surface, camera_offset)
    
    def get_sprites_at_position(self, pos, radius=5):
        """获取指定位置附近的精灵"""
        result = []
        test_rect = pygame.Rect(pos[0] - radius, pos[1] - radius, 
                               radius * 2, radius * 2)
        
        for sprite in self.all_sprites:
            if sprite.rect.colliderect(test_rect):
                result.append(sprite)
        
        return result
    
    def get_sprites_in_rect(self, rect):
        """获取矩形区域内的精灵"""
        result = []
        for sprite in self.all_sprites:
            if sprite.rect.colliderect(rect):
                result.append(sprite)
        return result
    
    def select_sprites(self, sprite_list):
        """选中指定的精灵"""
        # 清除之前的选择
        for sprite in self.selected_sprites:
            sprite.set_selected(False)
        self.selected_sprites.empty()
        
        # 选中新的精灵
        for sprite in sprite_list:
            sprite.set_selected(True)
            self.selected_sprites.add(sprite)
    
    def get_selected_entities(self):
        """获取选中的实体"""
        return [sprite.entity for sprite in self.selected_sprites]
    
    def check_collisions(self, group1_name, group2_name):
        """检查两个组之间的碰撞"""
        group1 = self.sprite_groups.get(group1_name) or self.collision_groups.get(group1_name)
        group2 = self.sprite_groups.get(group2_name) or self.collision_groups.get(group2_name)
        
        if not group1 or not group2:
            return []
        
        # 使用pygame的高效碰撞检测
        collisions = pygame.sprite.groupcollide(group1, group2, False, False)
        
        # 转换为实体对
        collision_pairs = []
        for sprite1, sprite_list in collisions.items():
            for sprite2 in sprite_list:
                collision_pairs.append((sprite1.entity, sprite2.entity))
        
        return collision_pairs
    
    def check_point_collisions(self, pos):
        """检查点与精灵的碰撞"""
        collided_sprites = []
        for sprite in self.all_sprites:
            if sprite.rect.collidepoint(pos):
                collided_sprites.append(sprite)
        return collided_sprites
    
    def get_sprite_by_entity_id(self, entity_id):
        """根据实体ID获取精灵"""
        return self.entity_sprite_map.get(entity_id)
    
    def get_sprites_by_type(self, entity_type):
        """根据类型获取精灵组"""
        type_map = {
            'ship': 'ships',
            'submarine': 'submarines',
            'bomber': 'aircraft',
            'missile': 'missiles',
            'ground_based_platforms': 'platforms',
            'airport': 'platforms',
        }
        
        group_name = type_map.get(entity_type)
        if group_name:
            return self.sprite_groups[group_name]
        return pygame.sprite.Group()
    
    def clear_all(self):
        """清空所有精灵"""
        self.all_sprites.empty()
        for group in self.sprite_groups.values():
            group.empty()
        for group in self.collision_groups.values():
            group.empty()
        self.selected_sprites.empty()
        self.entity_sprite_map.clear()
        self.sprite_factory.clear_cache()
    
    def get_statistics(self):
        """获取精灵管理统计信息"""
        return {
            'total_sprites': len(self.all_sprites),
            'selected_sprites': len(self.selected_sprites),
            'ships': len(self.sprite_groups['ships']),
            'submarines': len(self.sprite_groups['submarines']),
            'aircraft': len(self.sprite_groups['aircraft']),
            'missiles': len(self.sprite_groups['missiles']),
            'platforms': len(self.sprite_groups['platforms']),
            'friendly': len(self.collision_groups['friendly']),
            'enemy': len(self.collision_groups['enemy']),
            'neutral': len(self.collision_groups['neutral']),
        }