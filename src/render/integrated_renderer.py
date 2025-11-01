"""
集成渲染管理器
将分层渲染系统与现有的RenderManager整合
提供向后兼容性和渐进式优化
"""

import pygame
import time
from typing import Dict, List, Optional, Tuple, Any
from .layered_renderer import LayeredRenderManager, RenderLayer, ExplosionEffect, TrailEffect, LayeredRenderDebugger
from .sprite_manager import LayeredSpriteManager
from .optimized_loader import OptimizedSpriteLoader
from .single_process import RenderManager


class IntegratedRenderManager(RenderManager):
    """
    集成渲染管理器
    继承原有RenderManager，添加分层渲染功能
    """
    
    def __init__(self, env, screen_size, tile_size=64, terrain_override=None, 
                 show_obstacles=None, use_layered_rendering=True):
        # 初始化基础渲染管理器
        super().__init__(env, screen_size, tile_size, terrain_override, show_obstacles)
        
        # 分层渲染配置
        self.use_layered_rendering = use_layered_rendering
        
        if self.use_layered_rendering:
            # 初始化优化的资源加载器（使用更小的缓存限制）
            self.optimized_loader = OptimizedSpriteLoader(cache_limit=15)
            
            # 初始化优化的精灵管理器（需要sprite_loader）
            self.sprite_manager = LayeredSpriteManager(self.optimized_loader)
            
            # 初始化分层渲染系统
            self.layered_renderer = LayeredRenderManager(
                self.screen, 
                tuple(self.camera_offset)
            )
            
            # 初始化调试器
            self.debugger = LayeredRenderDebugger(self.layered_renderer)
            
            # 延迟预加载，只在需要时加载
            self._preload_enabled = False
            
            # 性能监控
            self.performance_monitor = PerformanceMonitor()
            
            # 渲染模式切换
            self.render_mode = 'layered'  # 'layered' 或 'legacy'
            
            # 预加载优化的精灵
            self._preload_optimized_sprites()
            
            # 实体同步状态
            self.entity_sync_cache = {}
            self.last_entity_count = 0
        else:
            self.layered_renderer = None
            self.sprite_manager = None
            self.optimized_loader = None
            self.debugger = None
            self.performance_monitor = None
            self.render_mode = 'legacy'
    
    def _preload_optimized_sprites(self):
        """预加载优化的精灵资源（按需加载）"""
        if not self.use_layered_rendering or not self._preload_enabled:
            return
        
        # 只预加载最常用的精灵类型
        essential_sprites = ['ship', 'submarine', 'missile']
        
        # 使用优化加载器批量加载
        self.optimized_loader.batch_load_sprites(essential_sprites)
        
        # 预加载地形（如果指定）
        if hasattr(self, 'terrain_override') and self.terrain_override:
            self.optimized_loader.load_terrain(
                image_file=self.terrain_override,
                size=(self.main_view_width, self.main_view_height)
            )
    
    def toggle_render_mode(self):
        """切换渲染模式"""
        if not self.use_layered_rendering:
            return
        
        if self.render_mode == 'layered':
            self.render_mode = 'legacy'
            print("切换到传统渲染模式")
        else:
            self.render_mode = 'layered'
            print("切换到分层渲染模式")
            # 重新同步所有实体
            self._sync_all_entities()
    
    def _sync_all_entities(self):
        """同步所有实体到分层渲染系统"""
        if not self.use_layered_rendering or self.render_mode != 'layered':
            return
        
        # 清空现有实体
        self.layered_renderer.entity_sprites.clear()
        for layer in RenderLayer:
            self.layered_renderer.layer_groups[layer].clear()
        self.layered_renderer.layered_group.empty()
        
        # 重新添加所有实体
        for player in self.env.players:
            for entity in player.entities:
                self.layered_renderer.add_entity(entity)
        
        # 更新同步缓存
        self.entity_sync_cache.clear()
        for player in self.env.players:
            for entity in player.entities:
                self.entity_sync_cache[entity.entity_id] = {
                    'position': getattr(entity, 'position', (0, 0)),
                    'rotation': getattr(entity, 'rotation', 0),
                    'health': getattr(entity, 'health', 100)
                }
    
    def _sync_entities_incremental(self):
        """增量同步实体变化"""
        if not self.use_layered_rendering or self.render_mode != 'layered':
            return
        
        current_entities = {}
        
        # 收集当前所有实体
        for player in self.env.players:
            for entity in player.entities:
                current_entities[entity.entity_id] = entity
        
        # 检查新增实体
        for entity_id, entity in current_entities.items():
            if entity_id not in self.layered_renderer.entity_sprites:
                self.layered_renderer.add_entity(entity)
                self.entity_sync_cache[entity_id] = {
                    'position': getattr(entity, 'position', (0, 0)),
                    'rotation': getattr(entity, 'rotation', 0),
                    'health': getattr(entity, 'health', 100)
                }
        
        # 检查移除的实体
        entities_to_remove = []
        for entity_id in self.layered_renderer.entity_sprites:
            if entity_id not in current_entities:
                entities_to_remove.append(entity_id)
        
        for entity_id in entities_to_remove:
            self.layered_renderer.remove_entity(entity_id)
            if entity_id in self.entity_sync_cache:
                del self.entity_sync_cache[entity_id]
        
        # 检查实体状态变化
        for entity_id, entity in current_entities.items():
            if entity_id in self.entity_sync_cache:
                cached_state = self.entity_sync_cache[entity_id]
                current_position = getattr(entity, 'position', (0, 0))
                current_rotation = getattr(entity, 'rotation', 0)
                current_health = getattr(entity, 'health', 100)
                
                # 检查是否有变化
                if (cached_state['position'] != current_position or
                    cached_state['rotation'] != current_rotation or
                    cached_state['health'] != current_health):
                    
                    # 更新缓存
                    self.entity_sync_cache[entity_id] = {
                        'position': current_position,
                        'rotation': current_rotation,
                        'health': current_health
                    }
                    
                    # 标记精灵需要更新
                    if entity_id in self.layered_renderer.entity_sprites:
                        sprite = self.layered_renderer.entity_sprites[entity_id]
                        sprite.needs_update = True
    
    def update(self):
        """更新渲染系统"""
        # 更新基础系统
        super().update()
        
        if self.use_layered_rendering and self.render_mode == 'layered':
            # 增量同步实体
            self._sync_entities_incremental()
            
            # 更新摄像机
            self.layered_renderer.update_camera(tuple(self.camera_offset))
            
            # 更新视口
            viewport = pygame.Rect(
                self.main_view_x, self.main_view_y,
                self.main_view_width, self.main_view_height
            )
            self.layered_renderer.update_viewport(viewport)
            
            # 更新分层渲染系统
            delta_time = self.clock.get_time() / 1000.0
            self.layered_renderer.update(delta_time)
            
            # 更新性能监控
            self.performance_monitor.update(delta_time)
    
    def draw_units(self):
        """绘制单位（支持两种渲染模式）"""
        if self.use_layered_rendering and self.render_mode == 'layered':
            self._draw_units_layered()
        else:
            # 使用传统渲染方式
            super().draw_units()
    
    def _draw_units_layered(self):
        """使用分层渲染系统绘制单位"""
        # 创建主视图剪裁区域
        main_view_rect = pygame.Rect(
            self.main_view_x, self.main_view_y,
            self.main_view_width, self.main_view_height
        )
        
        # 设置剪裁区域
        self.screen.set_clip(main_view_rect)
        
        # 创建主视图表面
        main_view_surface = pygame.Surface((self.main_view_width, self.main_view_height))
        
        # 渲染分层内容到主视图表面
        self.layered_renderer.render(main_view_surface)
        
        # 将主视图表面绘制到屏幕
        self.screen.blit(main_view_surface, (self.main_view_x, self.main_view_y))
        
        # 取消剪裁
        self.screen.set_clip(None)
        
        # 绘制选择框和UI覆盖层
        self._draw_selection_overlay()
    
    def _draw_selection_overlay(self):
        """绘制选择框和UI覆盖层"""
        # 绘制选择框
        if self.selection_rect:
            # 调整选择框到主视图坐标系
            adjusted_rect = pygame.Rect(
                self.selection_rect.x + self.main_view_x,
                self.selection_rect.y + self.main_view_y,
                self.selection_rect.width,
                self.selection_rect.height
            )
            pygame.draw.rect(self.screen, (0, 255, 0), adjusted_rect, 2)
        
        # 绘制选中实体的高亮
        for entity_id in self.selected_ids:
            if (self.use_layered_rendering and self.render_mode == 'layered' and
                entity_id in self.layered_renderer.entity_sprites):
                
                sprite = self.layered_renderer.entity_sprites[entity_id]
                highlight_rect = sprite.rect.copy()
                highlight_rect.x += self.camera_offset[0] + self.main_view_x
                highlight_rect.y += self.camera_offset[1] + self.main_view_y
                
                # 绘制高亮边框
                pygame.draw.rect(self.screen, (255, 255, 0), highlight_rect, 3)
    
    def add_explosion_effect(self, position: Tuple[float, float], size: float = 50.0):
        """添加爆炸特效"""
        if self.use_layered_rendering and self.render_mode == 'layered':
            effect = ExplosionEffect(position, size)
            self.layered_renderer.add_effect(effect)
    
    def add_trail_effect(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]):
        """添加轨迹特效"""
        if self.use_layered_rendering and self.render_mode == 'layered':
            effect = TrailEffect(start_pos, end_pos)
            self.layered_renderer.add_effect(effect)
    
    def handle_debug_keys(self, event):
        """处理调试按键"""
        if not self.use_layered_rendering:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1:
                # 切换渲染模式
                self.toggle_render_mode()
                return True
            elif event.key == pygame.K_F2:
                # 切换层信息显示
                self.debugger.toggle_layer_info()
                return True
            elif event.key == pygame.K_F3:
                # 切换性能信息显示
                self.debugger.toggle_performance()
                return True
            elif event.key == pygame.K_F4:
                # 切换边界显示
                self.debugger.toggle_bounds()
                return True
            elif event.key == pygame.K_F5:
                # 切换表面缓存
                self.layered_renderer.toggle_surface_cache()
                return True
            elif event.key == pygame.K_F6:
                # 切换视锥剔除
                self.layered_renderer.toggle_culling()
                return True
        
        return False
    
    def handle_events(self):
        """处理事件（扩展调试功能）"""
        events = pygame.event.get()
        
        for event in events:
            # 处理调试按键
            if self.handle_debug_keys(event):
                continue
            
            # 处理其他事件
            if event.type == pygame.QUIT:
                self.should_close = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.should_close = True
                elif event.key == pygame.K_h:
                    self.show_help_overlay = not self.show_help_overlay
                elif event.key == pygame.K_SPACE:
                    self.ui_actions.append('pause_toggle')
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.ui_actions.append('speed_up')
                elif event.key == pygame.K_MINUS:
                    self.ui_actions.append('speed_down')
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self._handle_left_click(event.pos)
                elif event.button == 3:  # 右键
                    self._handle_right_click(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.selecting:
                    self.selecting = False
                    self.selection_rect = None
            elif event.type == pygame.MOUSEMOTION:
                if self.selecting:
                    current_pos = event.pos
                    # 调整到主视图坐标系
                    if (self.main_view_x <= current_pos[0] <= self.main_view_x + self.main_view_width and
                        self.main_view_y <= current_pos[1] <= self.main_view_y + self.main_view_height):
                        
                        adjusted_current = (
                            current_pos[0] - self.main_view_x,
                            current_pos[1] - self.main_view_y
                        )
                        
                        self.selection_rect = pygame.Rect(
                            min(self.selection_start[0], adjusted_current[0]),
                            min(self.selection_start[1], adjusted_current[1]),
                            abs(adjusted_current[0] - self.selection_start[0]),
                            abs(adjusted_current[1] - self.selection_start[1])
                        )
        
        return events
    
    def draw_debug_info(self):
        """绘制调试信息"""
        if not self.use_layered_rendering:
            return
        
        # 绘制分层渲染调试信息
        self.debugger.render_debug_info(self.screen)
        
        # 绘制性能监控信息
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.render_performance_info(self.screen, self.font_small)
        
        # 绘制渲染模式信息
        mode_text = f"渲染模式: {self.render_mode}"
        mode_surface = self.font_small.render(mode_text, True, (255, 255, 255))
        self.screen.blit(mode_surface, (10, self.screen_height - 30))
    
    def render(self):
        """主渲染函数"""
        # 清空屏幕
        self.screen.fill((20, 30, 50))  # 深蓝色背景
        
        # 绘制地形
        self.draw_terrain()
        
        # 绘制单位
        self.draw_units()
        
        # 绘制UI
        self.draw_left_info_panel()
        self.draw_top_status_bar()
        self.draw_right_panel()
        self.draw_bottom_control_bar()
        self.draw_minimap()
        
        # 绘制调试信息
        self.draw_debug_info()
        
        # 绘制帮助覆盖层
        if self.show_help_overlay:
            self._draw_help_overlay()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_help_overlay(self):
        """绘制帮助覆盖层"""
        # 创建半透明覆盖层
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # 帮助文本
        help_lines = [
            "=== 海战指挥系统帮助 ===",
            "",
            "基本操作:",
            "  ESC - 退出游戏",
            "  H - 显示/隐藏帮助",
            "  空格 - 暂停/继续",
            "  +/- - 调整游戏速度",
            "",
            "调试功能 (需启用分层渲染):",
            "  F1 - 切换渲染模式",
            "  F2 - 显示层信息",
            "  F3 - 显示性能信息",
            "  F4 - 显示精灵边界",
            "  F5 - 切换表面缓存",
            "  F6 - 切换视锥剔除",
            "",
            "鼠标操作:",
            "  左键 - 选择单位",
            "  右键 - 移动/攻击命令",
            "  拖拽 - 框选多个单位"
        ]
        
        y_offset = 100
        for line in help_lines:
            if line.startswith("==="):
                color = (255, 255, 0)  # 黄色标题
                font = self.font_large
            elif line.startswith("  "):
                color = (200, 200, 200)  # 灰色内容
                font = self.font_small
            else:
                color = (255, 255, 255)  # 白色分类
                font = self.font
            
            if line.strip():  # 非空行
                text_surface = font.render(line, True, color)
                self.screen.blit(text_surface, (50, y_offset))
            
            y_offset += 25
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {
            'render_mode': self.render_mode,
            'use_layered_rendering': self.use_layered_rendering
        }
        
        if self.use_layered_rendering and self.layered_renderer:
            stats.update(self.layered_renderer.get_render_stats())
        
        if hasattr(self, 'performance_monitor'):
            stats.update(self.performance_monitor.get_stats())
        
        return stats


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.frame_times = []
        self.max_samples = 60  # 保存最近60帧的数据
        self.total_frames = 0
        self.start_time = time.time()
        
        # 性能指标
        self.current_fps = 0
        self.average_fps = 0
        self.min_fps = float('inf')
        self.max_fps = 0
        self.frame_time_ms = 0
        
        # 内存使用（如果可用）
        self.memory_usage = 0
        
        # 上一帧时间
        self.last_frame_time = time.time()
    
    def update(self, delta_time: float):
        """更新性能统计"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # 记录帧时间
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
        
        self.total_frames += 1
        
        # 计算FPS
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            self.current_fps = current_fps
            
            # 更新最值
            self.min_fps = min(self.min_fps, current_fps)
            self.max_fps = max(self.max_fps, current_fps)
            
            # 计算平均FPS
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.average_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            self.frame_time_ms = frame_time * 1000
        
        # 尝试获取内存使用情况
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_usage = 0
    
    def render_performance_info(self, surface: pygame.Surface, font: pygame.font.Font):
        """渲染性能信息"""
        info_lines = [
            f"FPS: {self.current_fps:.1f} (平均: {self.average_fps:.1f})",
            f"帧时间: {self.frame_time_ms:.2f}ms",
            f"最小/最大FPS: {self.min_fps:.1f}/{self.max_fps:.1f}",
            f"总帧数: {self.total_frames}"
        ]
        
        if self.memory_usage > 0:
            info_lines.append(f"内存: {self.memory_usage:.1f}MB")
        
        x_offset = surface.get_width() - 250
        y_offset = 50
        
        for line in info_lines:
            text_surface = font.render(line, True, (255, 255, 0))
            surface.blit(text_surface, (x_offset, y_offset))
            y_offset += 20
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计数据"""
        return {
            'current_fps': self.current_fps,
            'average_fps': self.average_fps,
            'min_fps': self.min_fps,
            'max_fps': self.max_fps,
            'frame_time_ms': self.frame_time_ms,
            'total_frames': self.total_frames,
            'memory_usage_mb': self.memory_usage
        }
    
    def reset_stats(self):
        """重置统计数据"""
        self.frame_times.clear()
        self.total_frames = 0
        self.start_time = time.time()
        self.min_fps = float('inf')
        self.max_fps = 0