"""
优化的碰撞检测系统
使用pygame内置碰撞检测功能替换手动实现
提供高效的空间分割和批量碰撞检测
"""

import pygame
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from .entities.entity import *
from .system_manager import System


class OptimizedCollisionSystem(System):
    """
    优化的碰撞检测系统
    使用pygame.sprite.Group和内置碰撞检测函数
    支持空间分割和层级碰撞检测
    """
    
    def __init__(self, game_data, game_map, event_manager, grid_size: int = 100):
        super().__init__(game_data)
        self.game_map = game_map
        self.event_manager = event_manager
        self.grid_size = grid_size
        
        # 碰撞组管理
        self.collision_groups: Dict[str, pygame.sprite.Group] = {
            'all': pygame.sprite.Group(),
            'ships': pygame.sprite.Group(),
            'submarines': pygame.sprite.Group(),
            'missiles': pygame.sprite.Group(),
            'aircraft': pygame.sprite.Group(),
            'ground_units': pygame.sprite.Group(),
            'friendly': pygame.sprite.Group(),
            'enemy': pygame.sprite.Group(),
            'neutral': pygame.sprite.Group()
        }
        
        # 空间分割网格
        self.spatial_grid: Dict[Tuple[int, int], Set[CollisionSprite]] = defaultdict(set)
        
        # 碰撞规则配置
        self.collision_rules = {
            'missiles': ['ships', 'submarines', 'aircraft', 'ground_units'],
            'ships': ['ships', 'submarines', 'ground_units'],
            'submarines': ['ships', 'submarines', 'ground_units'],
            'aircraft': ['aircraft', 'ground_units'],
            'ground_units': ['ships', 'submarines', 'aircraft', 'ground_units']
        }
        
        # 性能统计
        self.collision_stats = {
            'checks_per_frame': 0,
            'collisions_detected': 0,
            'spatial_grid_hits': 0,
            'pygame_collisions': 0
        }
        
        # 实体到精灵的映射
        self.entity_to_sprite: Dict[int, CollisionSprite] = {}
    
    def add_entity(self, entity) -> Optional['CollisionSprite']:
        """添加实体到碰撞检测系统"""
        position = entity.get_component(PositionComponent)
        collision = entity.get_component(CollisionComponent)
        
        if not (position and collision):
            return None
        
        # 创建碰撞精灵
        sprite = CollisionSprite(entity, position, collision)
        
        # 添加到相应的组
        self._add_to_groups(sprite, entity)
        
        # 更新空间网格
        self._update_spatial_grid(sprite)
        
        # 保存映射
        self.entity_to_sprite[entity.id] = sprite
        
        return sprite
    
    def remove_entity(self, entity_id: int):
        """从碰撞检测系统移除实体"""
        if entity_id in self.entity_to_sprite:
            sprite = self.entity_to_sprite[entity_id]
            
            # 从所有组中移除
            for group in self.collision_groups.values():
                group.remove(sprite)
            
            # 从空间网格移除
            self._remove_from_spatial_grid(sprite)
            
            # 清理映射
            del self.entity_to_sprite[entity_id]
    
    def update(self, delta_time):
        """更新碰撞检测"""
        self.collision_stats['checks_per_frame'] = 0
        self.collision_stats['collisions_detected'] = 0
        self.collision_stats['spatial_grid_hits'] = 0
        self.collision_stats['pygame_collisions'] = 0
        
        # 更新所有精灵位置
        self._update_sprite_positions()
        
        # 执行碰撞检测
        self._detect_collisions()
        
        # 检测地图碰撞
        self._detect_map_collisions()
    
    def _add_to_groups(self, sprite: 'CollisionSprite', entity):
        """将精灵添加到相应的组"""
        # 添加到全体组
        self.collision_groups['all'].add(sprite)
        
        # 根据实体类型添加到特定组
        entity_type = self._get_entity_type(entity)
        if entity_type in self.collision_groups:
            self.collision_groups[entity_type].add(sprite)
        
        # 根据阵营添加到阵营组
        faction = self._get_entity_faction(entity)
        if faction in self.collision_groups:
            self.collision_groups[faction].add(sprite)
    
    def _get_entity_type(self, entity) -> str:
        """获取实体类型"""
        # 这里需要根据实际的实体类型判断逻辑
        # 可以通过组件或实体类名来判断
        if hasattr(entity, 'entity_type'):
            return entity.entity_type
        
        # 根据组件判断
        if entity.get_component('ShipComponent'):
            return 'ships'
        elif entity.get_component('SubmarineComponent'):
            return 'submarines'
        elif entity.get_component('MissileComponent'):
            return 'missiles'
        elif entity.get_component('AircraftComponent'):
            return 'aircraft'
        elif entity.get_component('GroundUnitComponent'):
            return 'ground_units'
        
        return 'unknown'
    
    def _get_entity_faction(self, entity) -> str:
        """获取实体阵营"""
        if hasattr(entity, 'faction'):
            return entity.faction
        
        # 根据玩家ID判断
        if hasattr(entity, 'player_id'):
            if entity.player_id == 0:
                return 'friendly'
            elif entity.player_id == 1:
                return 'enemy'
        
        return 'neutral'
    
    def _update_sprite_positions(self):
        """更新所有精灵的位置"""
        for sprite in self.collision_groups['all']:
            old_grid_pos = sprite.grid_position
            sprite.update_position()
            
            # 如果网格位置改变，更新空间网格
            if sprite.grid_position != old_grid_pos:
                if old_grid_pos:
                    self.spatial_grid[old_grid_pos].discard(sprite)
                self.spatial_grid[sprite.grid_position].add(sprite)
    
    def _update_spatial_grid(self, sprite: 'CollisionSprite'):
        """更新精灵在空间网格中的位置"""
        grid_pos = sprite.grid_position
        self.spatial_grid[grid_pos].add(sprite)
    
    def _remove_from_spatial_grid(self, sprite: 'CollisionSprite'):
        """从空间网格移除精灵"""
        grid_pos = sprite.grid_position
        if grid_pos in self.spatial_grid:
            self.spatial_grid[grid_pos].discard(sprite)
    
    def _detect_collisions(self):
        """检测实体间碰撞"""
        # 使用空间分割优化碰撞检测
        checked_pairs = set()
        
        for grid_pos, sprites in self.spatial_grid.items():
            if len(sprites) < 2:
                continue
            
            # 检查相邻网格
            nearby_sprites = set(sprites)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                    nearby_sprites.update(self.spatial_grid.get(neighbor_pos, set()))
            
            # 在附近的精灵中检测碰撞
            sprites_list = list(nearby_sprites)
            for i, sprite1 in enumerate(sprites_list):
                for sprite2 in sprites_list[i+1:]:
                    pair = tuple(sorted([sprite1.entity.id, sprite2.entity.id]))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    
                    self.collision_stats['checks_per_frame'] += 1
                    
                    # 检查碰撞规则
                    if self._should_check_collision(sprite1, sprite2):
                        if self._check_collision(sprite1, sprite2):
                            self._handle_collision(sprite1, sprite2)
                            self.collision_stats['collisions_detected'] += 1
    
    def _should_check_collision(self, sprite1: 'CollisionSprite', sprite2: 'CollisionSprite') -> bool:
        """检查是否应该进行碰撞检测"""
        type1 = self._get_entity_type(sprite1.entity)
        type2 = self._get_entity_type(sprite2.entity)
        
        # 检查碰撞规则
        return (type2 in self.collision_rules.get(type1, []) or 
                type1 in self.collision_rules.get(type2, []))
    
    def _check_collision(self, sprite1: 'CollisionSprite', sprite2: 'CollisionSprite') -> bool:
        """使用pygame检测两个精灵的碰撞"""
        self.collision_stats['pygame_collisions'] += 1
        
        # 使用pygame的碰撞检测
        return pygame.sprite.collide_circle(sprite1, sprite2)
    
    def _handle_collision(self, sprite1: 'CollisionSprite', sprite2: 'CollisionSprite'):
        """处理碰撞事件"""
        # 触发碰撞事件
        if sprite1.collision_component.on_collide:
            sprite1.collision_component.on_collide(sprite2.entity)
        
        if sprite2.collision_component.on_collide:
            sprite2.collision_component.on_collide(sprite1.entity)
        
        # 发送碰撞事件
        if self.event_manager:
            from .event_manager import Event
            collision_event = Event('collision', {
                'entity1': sprite1.entity,
                'entity2': sprite2.entity,
                'position': sprite1.position_component.position
            })
            self.event_manager.publish(collision_event)
    
    def _detect_map_collisions(self):
        """检测与地图的碰撞"""
        for sprite in self.collision_groups['all']:
            pos = sprite.position_component.position
            x, y = int(pos[0]), int(pos[1])
            
            # 检查边界
            if (0 <= x < self.game_map.shape[0] and 
                0 <= y < self.game_map.shape[1]):
                
                if self.game_map[x, y] == 1:  # 障碍物
                    if sprite.collision_component.on_collide_map:
                        sprite.collision_component.on_collide_map()
    
    def get_collisions_in_area(self, center: Tuple[float, float], radius: float) -> List['CollisionSprite']:
        """获取指定区域内的所有碰撞体"""
        result = []
        
        # 计算涉及的网格范围
        grid_radius = int(radius / self.grid_size) + 1
        center_grid = (int(center[0] / self.grid_size), int(center[1] / self.grid_size))
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_pos = (center_grid[0] + dx, center_grid[1] + dy)
                
                for sprite in self.spatial_grid.get(grid_pos, set()):
                    pos = sprite.position_component.position
                    distance = np.linalg.norm(np.array(pos) - np.array(center))
                    if distance <= radius:
                        result.append(sprite)
        
        return result
    
    def get_collision_stats(self) -> Dict[str, Any]:
        """获取碰撞检测统计信息"""
        return {
            'total_entities': len(self.entity_to_sprite),
            'spatial_grid_cells': len(self.spatial_grid),
            'performance': self.collision_stats.copy(),
            'group_sizes': {name: len(group) for name, group in self.collision_groups.items()}
        }


class CollisionSprite(pygame.sprite.Sprite):
    """
    碰撞检测精灵
    将ECS实体适配为pygame精灵用于碰撞检测
    """
    
    def __init__(self, entity, position_component, collision_component, grid_size: int = 100):
        super().__init__()
        
        self.entity = entity
        self.position_component = position_component
        self.collision_component = collision_component
        self.grid_size = grid_size
        
        # 创建碰撞矩形
        radius = collision_component.radius
        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        
        # 设置碰撞半径（用于circle collision）
        self.radius = radius
        
        # 初始化位置
        self.update_position()
        
        # 网格位置缓存
        self.grid_position = self._calculate_grid_position()
    
    def update_position(self):
        """更新精灵位置"""
        pos = self.position_component.position
        self.rect.centerx = int(pos[0])
        self.rect.centery = int(pos[1])
        self.grid_position = self._calculate_grid_position()
    
    def _calculate_grid_position(self) -> Tuple[int, int]:
        """计算网格位置"""
        return (
            int(self.rect.centerx / self.grid_size),
            int(self.rect.centery / self.grid_size)
        )


class CollisionLayer:
    """
    碰撞层管理器
    支持分层碰撞检测，提高性能
    """
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.group = pygame.sprite.Group()
        self.enabled = True
    
    def add(self, sprite: CollisionSprite):
        """添加精灵到层"""
        self.group.add(sprite)
    
    def remove(self, sprite: CollisionSprite):
        """从层移除精灵"""
        self.group.remove(sprite)
    
    def update(self):
        """更新层中的所有精灵"""
        if self.enabled:
            self.group.update()
    
    def check_collisions(self, other_layer: 'CollisionLayer') -> List[Tuple[CollisionSprite, CollisionSprite]]:
        """检查与另一层的碰撞"""
        if not (self.enabled and other_layer.enabled):
            return []
        
        collisions = []
        collision_dict = pygame.sprite.groupcollide(
            self.group, other_layer.group, False, False,
            pygame.sprite.collide_circle
        )
        
        for sprite1, sprite_list in collision_dict.items():
            for sprite2 in sprite_list:
                collisions.append((sprite1, sprite2))
        
        return collisions


class LayeredCollisionManager:
    """
    分层碰撞管理器
    管理多个碰撞层，支持层间碰撞检测
    """
    
    def __init__(self):
        self.layers: Dict[str, CollisionLayer] = {}
        self.collision_matrix: Dict[Tuple[str, str], bool] = {}
    
    def add_layer(self, name: str, priority: int = 0) -> CollisionLayer:
        """添加碰撞层"""
        layer = CollisionLayer(name, priority)
        self.layers[name] = layer
        return layer
    
    def set_layer_collision(self, layer1: str, layer2: str, enabled: bool = True):
        """设置两层之间是否检测碰撞"""
        key = tuple(sorted([layer1, layer2]))
        self.collision_matrix[key] = enabled
    
    def update(self) -> List[Tuple[CollisionSprite, CollisionSprite]]:
        """更新所有层并检测碰撞"""
        # 更新所有层
        for layer in self.layers.values():
            layer.update()
        
        # 检测层间碰撞
        all_collisions = []
        layer_names = list(self.layers.keys())
        
        for i, name1 in enumerate(layer_names):
            for name2 in layer_names[i:]:
                key = tuple(sorted([name1, name2]))
                
                if self.collision_matrix.get(key, True):
                    layer1 = self.layers[name1]
                    layer2 = self.layers[name2]
                    
                    if name1 == name2:
                        # 同层内碰撞检测
                        collisions = self._check_internal_collisions(layer1)
                    else:
                        # 跨层碰撞检测
                        collisions = layer1.check_collisions(layer2)
                    
                    all_collisions.extend(collisions)
        
        return all_collisions
    
    def _check_internal_collisions(self, layer: CollisionLayer) -> List[Tuple[CollisionSprite, CollisionSprite]]:
        """检查层内碰撞"""
        collisions = []
        sprites = list(layer.group.sprites())
        
        for i, sprite1 in enumerate(sprites):
            for sprite2 in sprites[i+1:]:
                if pygame.sprite.collide_circle(sprite1, sprite2):
                    collisions.append((sprite1, sprite2))
        
        return collisions