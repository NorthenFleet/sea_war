"""
碰撞检测系统集成
将优化的pygame碰撞检测系统集成到现有的ECS架构中
提供向后兼容的接口和渐进式迁移支持
"""

import pygame
import numpy as np
from typing import Dict, List, Optional, Any, Union
from .optimized_collision import OptimizedCollisionSystem, CollisionSprite
from .system_manager import CollisionSystem as OriginalCollisionSystem
from .entities.entity import *
from .event_manager import EventManager


class HybridCollisionSystem:
    """
    混合碰撞检测系统
    支持原有系统和优化系统的并行运行
    提供渐进式迁移路径
    """
    
    def __init__(self, game_data, game_map, event_manager: EventManager, 
                 use_optimized: bool = True, fallback_enabled: bool = True):
        self.game_data = game_data
        self.game_map = game_map
        self.event_manager = event_manager
        self.use_optimized = use_optimized
        self.fallback_enabled = fallback_enabled
        
        # 初始化两套系统
        if use_optimized:
            self.optimized_system = OptimizedCollisionSystem(
                game_data, game_map, event_manager
            )
        else:
            self.optimized_system = None
        
        if fallback_enabled:
            self.original_system = OriginalCollisionSystem(game_data, game_map)
        else:
            self.original_system = None
        
        # 性能对比数据
        self.performance_comparison = {
            'optimized': {'frame_time': [], 'collision_count': []},
            'original': {'frame_time': [], 'collision_count': []},
            'current_mode': 'optimized' if use_optimized else 'original'
        }
        
        # 实体注册表
        self.registered_entities = set()
    
    def add_entity(self, entity):
        """添加实体到碰撞检测系统"""
        if entity.id in self.registered_entities:
            return
        
        self.registered_entities.add(entity.id)
        
        if self.optimized_system:
            self.optimized_system.add_entity(entity)
    
    def remove_entity(self, entity_id: int):
        """从碰撞检测系统移除实体"""
        if entity_id not in self.registered_entities:
            return
        
        self.registered_entities.discard(entity_id)
        
        if self.optimized_system:
            self.optimized_system.remove_entity(entity_id)
    
    def update(self, delta_time):
        """更新碰撞检测系统"""
        import time
        
        if self.use_optimized and self.optimized_system:
            # 使用优化系统
            start_time = time.time()
            self.optimized_system.update(delta_time)
            frame_time = time.time() - start_time
            
            stats = self.optimized_system.get_collision_stats()
            self.performance_comparison['optimized']['frame_time'].append(frame_time)
            self.performance_comparison['optimized']['collision_count'].append(
                stats['performance']['collisions_detected']
            )
            
        elif self.original_system:
            # 使用原始系统
            start_time = time.time()
            self.original_system.update(delta_time)
            frame_time = time.time() - start_time
            
            self.performance_comparison['original']['frame_time'].append(frame_time)
            # 原始系统没有统计碰撞数量，使用估算值
            self.performance_comparison['original']['collision_count'].append(0)
        
        # 保持性能数据在合理范围内
        for mode_data in self.performance_comparison.values():
            if isinstance(mode_data, dict):
                for key in ['frame_time', 'collision_count']:
                    if key in mode_data and len(mode_data[key]) > 1000:
                        mode_data[key] = mode_data[key][-500:]  # 保留最近500帧数据
    
    def switch_mode(self, use_optimized: bool):
        """切换碰撞检测模式"""
        if use_optimized == self.use_optimized:
            return
        
        self.use_optimized = use_optimized
        self.performance_comparison['current_mode'] = 'optimized' if use_optimized else 'original'
        
        # 如果切换到优化模式但系统未初始化，则初始化
        if use_optimized and not self.optimized_system:
            self.optimized_system = OptimizedCollisionSystem(
                self.game_data, self.game_map, self.event_manager
            )
            # 重新注册所有实体
            for entity in self.game_data.get_all_entities():
                if entity.id in self.registered_entities:
                    self.optimized_system.add_entity(entity)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能对比报告"""
        report = {
            'current_mode': self.performance_comparison['current_mode'],
            'optimized_available': self.optimized_system is not None,
            'original_available': self.original_system is not None,
            'registered_entities': len(self.registered_entities)
        }
        
        # 计算性能统计
        for mode in ['optimized', 'original']:
            data = self.performance_comparison[mode]
            if data['frame_time']:
                report[f'{mode}_stats'] = {
                    'avg_frame_time': np.mean(data['frame_time']),
                    'max_frame_time': np.max(data['frame_time']),
                    'min_frame_time': np.min(data['frame_time']),
                    'total_frames': len(data['frame_time']),
                    'avg_collisions': np.mean(data['collision_count']) if data['collision_count'] else 0
                }
            else:
                report[f'{mode}_stats'] = None
        
        # 添加优化系统的详细统计
        if self.optimized_system:
            report['optimized_details'] = self.optimized_system.get_collision_stats()
        
        return report
    
    def get_collisions_in_area(self, center, radius):
        """获取指定区域内的碰撞"""
        if self.use_optimized and self.optimized_system:
            return self.optimized_system.get_collisions_in_area(center, radius)
        else:
            # 原始系统的简单实现
            result = []
            for entity in self.game_data.get_all_entities():
                position = entity.get_component(PositionComponent)
                collision = entity.get_component(CollisionComponent)
                if position and collision:
                    distance = np.linalg.norm(np.array(position.position) - np.array(center))
                    if distance <= radius:
                        result.append(entity)
            return result


class CollisionSystemAdapter:
    """
    碰撞系统适配器
    为现有代码提供统一的接口，无需修改现有调用代码
    """
    
    def __init__(self, hybrid_system: HybridCollisionSystem):
        self.hybrid_system = hybrid_system
    
    def update(self, delta_time):
        """兼容原始CollisionSystem的update接口"""
        self.hybrid_system.update(delta_time)
    
    def add_entity(self, entity):
        """添加实体"""
        self.hybrid_system.add_entity(entity)
    
    def remove_entity(self, entity_id):
        """移除实体"""
        self.hybrid_system.remove_entity(entity_id)


class CollisionSystemFactory:
    """
    碰撞系统工厂
    根据配置创建合适的碰撞检测系统
    """
    
    @staticmethod
    def create_collision_system(game_data, game_map, event_manager, 
                              config: Dict[str, Any] = None):
        """
        创建碰撞检测系统
        
        Args:
            game_data: 游戏数据
            game_map: 游戏地图
            event_manager: 事件管理器
            config: 配置字典
        
        Returns:
            碰撞检测系统实例
        """
        if config is None:
            config = {}
        
        # 默认配置
        default_config = {
            'use_optimized': True,
            'fallback_enabled': True,
            'auto_switch': False,
            'performance_threshold': 0.016,  # 60fps阈值
            'grid_size': 100
        }
        
        # 合并配置
        final_config = {**default_config, **config}
        
        # 创建混合系统
        hybrid_system = HybridCollisionSystem(
            game_data, game_map, event_manager,
            use_optimized=final_config['use_optimized'],
            fallback_enabled=final_config['fallback_enabled']
        )
        
        # 如果启用自动切换，创建自适应系统
        if final_config['auto_switch']:
            return AdaptiveCollisionSystem(hybrid_system, final_config)
        else:
            return CollisionSystemAdapter(hybrid_system)


class AdaptiveCollisionSystem(CollisionSystemAdapter):
    """
    自适应碰撞检测系统
    根据性能自动在优化系统和原始系统间切换
    """
    
    def __init__(self, hybrid_system: HybridCollisionSystem, config: Dict[str, Any]):
        super().__init__(hybrid_system)
        self.config = config
        self.performance_threshold = config['performance_threshold']
        self.check_interval = 60  # 每60帧检查一次性能
        self.frame_count = 0
        self.last_switch_frame = 0
        self.min_switch_interval = 300  # 最小切换间隔300帧
    
    def update(self, delta_time):
        """更新并监控性能"""
        super().update(delta_time)
        
        self.frame_count += 1
        
        # 定期检查性能
        if (self.frame_count % self.check_interval == 0 and 
            self.frame_count - self.last_switch_frame > self.min_switch_interval):
            self._check_and_switch_if_needed()
    
    def _check_and_switch_if_needed(self):
        """检查性能并在需要时切换系统"""
        report = self.hybrid_system.get_performance_report()
        current_mode = report['current_mode']
        
        # 获取当前模式的性能数据
        current_stats = report.get(f'{current_mode}_stats')
        if not current_stats:
            return
        
        avg_frame_time = current_stats['avg_frame_time']
        
        # 如果当前是优化模式但性能不佳，切换到原始模式
        if (current_mode == 'optimized' and 
            avg_frame_time > self.performance_threshold and
            report['original_available']):
            
            print(f"Performance degraded (avg: {avg_frame_time:.4f}s), switching to original collision system")
            self.hybrid_system.switch_mode(False)
            self.last_switch_frame = self.frame_count
        
        # 如果当前是原始模式且优化系统可用，尝试切换回优化模式
        elif (current_mode == 'original' and 
              report['optimized_available']):
            
            print("Attempting to switch back to optimized collision system")
            self.hybrid_system.switch_mode(True)
            self.last_switch_frame = self.frame_count


def migrate_to_optimized_collision(sea_war_env):
    """
    将现有的SeaWarEnv迁移到优化的碰撞检测系统
    
    Args:
        sea_war_env: SeaWarEnv实例
    
    Returns:
        更新后的系统列表
    """
    # 创建优化的碰撞检测系统
    collision_system = CollisionSystemFactory.create_collision_system(
        sea_war_env.game_data,
        sea_war_env.game_map,
        sea_war_env.event_manager,
        config={
            'use_optimized': True,
            'fallback_enabled': True,
            'auto_switch': True
        }
    )
    
    # 将碰撞系统添加到系统列表
    sea_war_env.collision_system = collision_system
    sea_war_env.systems.append(collision_system)
    
    # 注册现有实体
    for entity in sea_war_env.game_data.get_all_entities():
        collision_system.add_entity(entity)
    
    return sea_war_env.systems


class CollisionDebugger:
    """
    碰撞检测调试器
    提供可视化和性能分析工具
    """
    
    def __init__(self, collision_system):
        self.collision_system = collision_system
        self.debug_surface = None
        self.show_collision_bounds = False
        self.show_spatial_grid = False
        self.show_performance_overlay = False
    
    def toggle_collision_bounds(self):
        """切换碰撞边界显示"""
        self.show_collision_bounds = not self.show_collision_bounds
    
    def toggle_spatial_grid(self):
        """切换空间网格显示"""
        self.show_spatial_grid = not self.show_spatial_grid
    
    def toggle_performance_overlay(self):
        """切换性能覆盖显示"""
        self.show_performance_overlay = not self.show_performance_overlay
    
    def render_debug_info(self, screen):
        """渲染调试信息"""
        if not isinstance(self.collision_system, CollisionSystemAdapter):
            return
        
        hybrid_system = self.collision_system.hybrid_system
        
        if self.show_collision_bounds and hybrid_system.optimized_system:
            self._draw_collision_bounds(screen, hybrid_system.optimized_system)
        
        if self.show_spatial_grid and hybrid_system.optimized_system:
            self._draw_spatial_grid(screen, hybrid_system.optimized_system)
        
        if self.show_performance_overlay:
            self._draw_performance_overlay(screen, hybrid_system)
    
    def _draw_collision_bounds(self, screen, optimized_system):
        """绘制碰撞边界"""
        for sprite in optimized_system.collision_groups['all']:
            pygame.draw.circle(screen, (255, 0, 0), 
                             (sprite.rect.centerx, sprite.rect.centery), 
                             sprite.radius, 1)
    
    def _draw_spatial_grid(self, screen, optimized_system):
        """绘制空间网格"""
        grid_size = optimized_system.grid_size
        screen_width, screen_height = screen.get_size()
        
        # 绘制网格线
        for x in range(0, screen_width, grid_size):
            pygame.draw.line(screen, (100, 100, 100), (x, 0), (x, screen_height))
        
        for y in range(0, screen_height, grid_size):
            pygame.draw.line(screen, (100, 100, 100), (0, y), (screen_width, y))
        
        # 高亮有实体的网格
        for (gx, gy), sprites in optimized_system.spatial_grid.items():
            if sprites:
                rect = pygame.Rect(gx * grid_size, gy * grid_size, grid_size, grid_size)
                pygame.draw.rect(screen, (0, 255, 0, 50), rect)
    
    def _draw_performance_overlay(self, screen, hybrid_system):
        """绘制性能覆盖信息"""
        font = pygame.font.Font(None, 24)
        report = hybrid_system.get_performance_report()
        
        y_offset = 10
        lines = [
            f"Mode: {report['current_mode']}",
            f"Entities: {report['registered_entities']}"
        ]
        
        current_stats = report.get(f"{report['current_mode']}_stats")
        if current_stats:
            lines.extend([
                f"Avg Frame Time: {current_stats['avg_frame_time']:.4f}s",
                f"Max Frame Time: {current_stats['max_frame_time']:.4f}s",
                f"Avg Collisions: {current_stats['avg_collisions']:.1f}"
            ])
        
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, y_offset))
            y_offset += 25