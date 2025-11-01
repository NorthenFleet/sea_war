"""
优化的图像资源加载器
使用pygame的convert()和convert_alpha()方法优化渲染性能
提供缓存机制和批量加载功能
"""

import pygame
import os
import json
import subprocess
import tempfile
from typing import Dict, Optional, Tuple, Any
from pathlib import Path


class OptimizedSpriteLoader:
    """
    优化的精灵加载器
    自动应用convert()和convert_alpha()优化
    支持缓存、批量加载和资源管理
    """
    
    def __init__(self, images_dir: str = None, map_dir: str = None, cache_limit: int = 20):
        # 目录设置
        if images_dir is None:
            images_dir = os.path.join(os.path.dirname(__file__), 'images')
        if map_dir is None:
            map_dir = os.path.join(os.path.dirname(__file__), 'map')
        
        self.images_dir = Path(images_dir)
        self.map_dir = Path(map_dir)
        
        # 确保目录存在
        self.images_dir.mkdir(exist_ok=True)
        self.map_dir.mkdir(exist_ok=True)
        
        # 缓存（更小的缓存限制）
        self.sprite_cache: Dict[str, pygame.Surface] = {}
        self.metadata_cache: Dict[str, Dict] = {}
        self.cache_limit = cache_limit
        self.cache_access_order = []  # LRU缓存顺序
        
        # 启用内存优化模式
        self.memory_optimization = True
        
        # 默认配置
        self.default_sizes = {
            'ship': (36, 18),
            'submarine': (30, 14),
            'missile': (8, 20),
            'ground_based_platforms': (24, 24),
            'airport': (48, 32),
            'bomber': (40, 40)
        }
        
        self.default_colors = {
            'ship': (70, 130, 180),
            'submarine': (72, 61, 139),
            'missile': (220, 20, 60),
            'ground_based_platforms': (255, 215, 0),
            'airport': (105, 105, 105),
            'bomber': (255, 140, 0)
        }
        
        # 支持的图像格式（包括SVG）
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.gif', '.svg'}
        
        # 性能统计
        self.load_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'conversions_applied': 0,
            'placeholder_created': 0
        }
    
    def load_sprite(self, sprite_name: str, target_size: Optional[Tuple[int, int]] = None, 
                   force_reload: bool = False) -> pygame.Surface:
        """
        加载并优化单个精灵
        
        Args:
            sprite_name: 精灵名称
            target_size: 目标尺寸，None表示使用默认尺寸
            force_reload: 是否强制重新加载
        
        Returns:
            优化后的pygame.Surface
        """
        cache_key = f"{sprite_name}_{target_size}"
        
        # 检查缓存
        if not force_reload and cache_key in self.sprite_cache:
            self.load_stats['cache_hits'] += 1
            # 更新LRU顺序
            self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            return self.sprite_cache[cache_key]
        
        self.load_stats['cache_misses'] += 1
        
        # 查找图像文件
        image_path = self._find_image_file(sprite_name)
        
        if image_path and image_path.exists():
            surface = self._load_and_optimize_image(image_path, target_size)
        else:
            # 创建占位图像
            surface = self._create_optimized_placeholder(sprite_name, target_size)
            self.load_stats['placeholder_created'] += 1
        
        # 缓存结果
        self._add_to_cache(cache_key, surface)
        return surface
    
    def _add_to_cache(self, cache_key: str, surface: pygame.Surface):
        """添加到缓存，支持LRU淘汰策略"""
        # 如果缓存已满，移除最久未使用的项
        if len(self.sprite_cache) >= self.cache_limit:
            if self.cache_access_order:
                oldest_key = self.cache_access_order.pop(0)
                if oldest_key in self.sprite_cache:
                    del self.sprite_cache[oldest_key]
        
        # 添加新项
        self.sprite_cache[cache_key] = surface
        self.cache_access_order.append(cache_key)
    
    def _find_image_file(self, sprite_name: str) -> Optional[Path]:
        """查找图像文件"""
        # 直接匹配
        for ext in self.supported_formats:
            path = self.images_dir / f"{sprite_name}{ext}"
            if path.exists():
                return path
        
        # 模糊匹配（处理不同命名约定）
        name_variants = [
            sprite_name,
            sprite_name.lower(),
            sprite_name.upper(),
            sprite_name.replace('_', '-'),
            sprite_name.replace('-', '_')
        ]
        
        for variant in name_variants:
            for ext in self.supported_formats:
                path = self.images_dir / f"{variant}{ext}"
                if path.exists():
                    return path
        
        return None
    
    def _load_and_optimize_image(self, image_path: Path, 
                                target_size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
        """加载并优化图像"""
        try:
            # 处理SVG文件
            if image_path.suffix.lower() == '.svg':
                return self._load_svg_image(image_path, target_size)
            
            # 加载原始图像
            surface = pygame.image.load(str(image_path))
            
            # 在内存优化模式下，优先使用convert()而不是convert_alpha()
            if self.memory_optimization:
                # 检查是否真的需要alpha通道
                has_alpha = surface.get_flags() & pygame.SRCALPHA or surface.get_colorkey() is not None
                if has_alpha and surface.get_bitsize() > 24:
                    surface = surface.convert_alpha()
                else:
                    surface = surface.convert()
            else:
                # 原始逻辑
                has_alpha = surface.get_flags() & pygame.SRCALPHA or surface.get_colorkey() is not None
                if has_alpha:
                    surface = surface.convert_alpha()
                else:
                    surface = surface.convert()
            
            self.load_stats['conversions_applied'] += 1
            
            # 调整尺寸
            if target_size:
                surface = pygame.transform.smoothscale(surface, target_size)
            elif image_path.stem in self.default_sizes:
                default_size = self.default_sizes[image_path.stem]
                if surface.get_size() != default_size:
                    surface = pygame.transform.smoothscale(surface, default_size)
            
            return surface
            
        except pygame.error as e:
            print(f"Failed to load image {image_path}: {e}")
            # 返回占位图像
            sprite_name = image_path.stem
            return self._create_optimized_placeholder(sprite_name, target_size)
    
    def _load_svg_image(self, svg_path: Path, target_size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
        """加载SVG图像并转换为pygame Surface"""
        try:
            # 确定目标尺寸
            if target_size is None and svg_path.stem in self.default_sizes:
                target_size = self.default_sizes[svg_path.stem]
            elif target_size is None:
                target_size = (32, 32)  # 默认尺寸
            
            # 尝试使用cairosvg（如果可用）
            try:
                import cairosvg
                import io
                from PIL import Image
                
                # 将SVG转换为PNG字节流
                png_data = cairosvg.svg2png(
                    url=str(svg_path),
                    output_width=target_size[0],
                    output_height=target_size[1]
                )
                
                # 使用PIL加载PNG数据
                pil_image = Image.open(io.BytesIO(png_data))
                
                # 转换为pygame Surface
                mode = pil_image.mode
                size = pil_image.size
                raw = pil_image.tobytes()
                
                surface = pygame.image.fromstring(raw, size, mode)
                if self.memory_optimization:
                    surface = surface.convert_alpha()
                else:
                    surface = surface.convert_alpha()
                
                return surface
                
            except ImportError:
                # 如果cairosvg不可用，尝试使用系统命令
                return self._load_svg_with_system_command(svg_path, target_size)
                
        except Exception as e:
            print(f"Failed to load SVG {svg_path}: {e}")
            # 返回占位图像
            return self._create_optimized_placeholder(svg_path.stem, target_size)
    
    def _load_svg_with_system_command(self, svg_path: Path, target_size: Tuple[int, int]) -> pygame.Surface:
        """使用系统命令加载SVG（备用方法）"""
        try:
            # 创建临时PNG文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_png_path = temp_file.name
            
            # 尝试使用rsvg-convert（如果可用）
            try:
                subprocess.run([
                    'rsvg-convert',
                    '-w', str(target_size[0]),
                    '-h', str(target_size[1]),
                    '-o', temp_png_path,
                    str(svg_path)
                ], check=True, capture_output=True)
                
                # 加载转换后的PNG
                surface = pygame.image.load(temp_png_path)
                surface = surface.convert_alpha()
                
                # 清理临时文件
                os.unlink(temp_png_path)
                
                return surface
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                # 如果rsvg-convert不可用，创建简单的占位图像
                os.unlink(temp_png_path)  # 清理临时文件
                return self._create_svg_placeholder(svg_path.stem, target_size)
                
        except Exception as e:
            print(f"System command SVG loading failed for {svg_path}: {e}")
            return self._create_svg_placeholder(svg_path.stem, target_size)
    
    def _create_svg_placeholder(self, sprite_name: str, target_size: Tuple[int, int]) -> pygame.Surface:
        """为SVG文件创建占位图像"""
        surface = pygame.Surface(target_size, pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))  # 透明背景
        
        # 根据精灵类型创建简单的形状
        color = self.default_colors.get(sprite_name, (128, 128, 128))
        
        if 'ship' in sprite_name:
            # 绘制船形
            points = [
                (target_size[0] * 0.1, target_size[1] * 0.7),
                (target_size[0] * 0.9, target_size[1] * 0.7),
                (target_size[0] * 0.8, target_size[1] * 0.3),
                (target_size[0] * 0.2, target_size[1] * 0.3)
            ]
            pygame.draw.polygon(surface, color, points)
        elif 'submarine' in sprite_name:
            # 绘制潜艇形
            pygame.draw.ellipse(surface, color, (
                target_size[0] * 0.1, target_size[1] * 0.3,
                target_size[0] * 0.8, target_size[1] * 0.4
            ))
        elif 'missile' in sprite_name:
            # 绘制导弹形
            pygame.draw.rect(surface, color, (
                target_size[0] * 0.3, target_size[1] * 0.1,
                target_size[0] * 0.4, target_size[1] * 0.8
            ))
        else:
            # 默认矩形
            pygame.draw.rect(surface, color, (
                target_size[0] * 0.2, target_size[1] * 0.2,
                target_size[0] * 0.6, target_size[1] * 0.6
            ))
        
        return surface
    
    def _create_optimized_placeholder(self, sprite_name: str, 
                                    target_size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
        """创建优化的占位图像"""
        # 确定尺寸
        if target_size:
            size = target_size
        else:
            size = self.default_sizes.get(sprite_name, (28, 28))
        
        # 确定颜色
        color = self.default_colors.get(sprite_name, (200, 200, 200))
        
        # 创建表面
        surface = pygame.Surface(size, pygame.SRCALPHA)
        surface.fill(color)
        
        # 添加形状标识
        self._add_shape_identifier(surface, sprite_name, size)
        
        # 优化表面
        surface = surface.convert_alpha()
        
        return surface
    
    def _add_shape_identifier(self, surface: pygame.Surface, sprite_name: str, size: Tuple[int, int]):
        """为占位图像添加形状标识"""
        w, h = size
        
        if sprite_name in ('ship', 'submarine'):
            # 绘制矩形边框
            pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 1)
            # 添加方向指示
            pygame.draw.polygon(surface, (255, 255, 255), 
                              [(w-5, h//2-3), (w-1, h//2), (w-5, h//2+3)])
        
        elif sprite_name == 'missile':
            # 绘制导弹形状
            pygame.draw.polygon(surface, (255, 255, 255), 
                              [(w//2, 0), (w, h//2), (w//2, h), (0, h//2)])
        
        elif sprite_name == 'bomber':
            # 绘制飞机形状
            pygame.draw.ellipse(surface, (255, 255, 255), 
                              (w//4, h//4, w//2, h//2))
            pygame.draw.rect(surface, (255, 255, 255), 
                           (w//2-2, 0, 4, h))
        
        elif sprite_name in ('ground_based_platforms', 'airport'):
            # 绘制方形结构
            pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 2)
            pygame.draw.rect(surface, (255, 255, 255), 
                           (w//4, h//4, w//2, h//2))
    
    def load_terrain(self, terrain_name: str = None, screen_size: Tuple[int, int] = (1024, 768)) -> pygame.Surface:
        """
        加载地形图像
        
        Args:
            terrain_name: 地形文件名，None表示自动查找
            screen_size: 屏幕尺寸，用于缩放地形
        
        Returns:
            优化后的地形Surface
        """
        cache_key = f"terrain_{terrain_name}_{screen_size}"
        
        if cache_key in self.sprite_cache:
            self.load_stats['cache_hits'] += 1
            # 更新LRU顺序
            self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            return self.sprite_cache[cache_key]
        
        self.load_stats['cache_misses'] += 1
        
        # 处理纯色地形
        if terrain_name and terrain_name.startswith('color:'):
            surface = self._create_color_terrain(terrain_name, screen_size)
        else:
            # 查找地形文件
            terrain_path = self._find_terrain_file(terrain_name)
            if terrain_path:
                surface = self._load_terrain_image(terrain_path, screen_size)
            else:
                # 创建默认地形
                surface = self._create_default_terrain(screen_size)
        
        # 缓存结果
        self._add_to_cache(cache_key, surface)
        return surface
    
    def _find_terrain_file(self, terrain_name: Optional[str]) -> Optional[Path]:
        """查找地形文件"""
        candidates = []
        
        if terrain_name:
            candidates.append(terrain_name)
        
        # 默认候选文件
        candidates.extend([
            '六角格地图.jpg',
            'map.png', 'map.jpg', 'terrain.png', 'terrain.jpg',
            'ground.png', '地图.png', '地图.jpg'
        ])
        
        for candidate in candidates:
            path = self.map_dir / candidate
            if path.exists():
                return path
            
            # 递归搜索子目录
            for subpath in self.map_dir.rglob(candidate):
                if subpath.is_file():
                    return subpath
        
        return None
    
    def _load_terrain_image(self, terrain_path: Path, screen_size: Tuple[int, int]) -> pygame.Surface:
        """加载地形图像"""
        try:
            surface = pygame.image.load(str(terrain_path))
            
            # 缩放到屏幕尺寸
            surface = pygame.transform.smoothscale(surface, screen_size)
            
            # 优化表面
            surface = surface.convert()
            self.load_stats['conversions_applied'] += 1
            
            return surface
            
        except pygame.error as e:
            print(f"Failed to load terrain {terrain_path}: {e}")
            return self._create_default_terrain(screen_size)
    
    def _create_color_terrain(self, color_spec: str, screen_size: Tuple[int, int]) -> pygame.Surface:
        """创建纯色地形"""
        color_part = color_spec.split(':', 1)[1].strip()
        color = (0, 100, 200)  # 默认海洋蓝
        
        try:
            if color_part.startswith('#'):
                # 十六进制颜色
                if len(color_part) == 4:  # #RGB
                    r = int(color_part[1] * 2, 16)
                    g = int(color_part[2] * 2, 16)
                    b = int(color_part[3] * 2, 16)
                else:  # #RRGGBB
                    r = int(color_part[1:3], 16)
                    g = int(color_part[3:5], 16)
                    b = int(color_part[5:7], 16)
                color = (r, g, b)
            else:
                # RGB数值
                parts = [int(x.strip()) for x in color_part.split(',')]
                if len(parts) == 3:
                    color = tuple(max(0, min(255, v)) for v in parts)
        except (ValueError, IndexError):
            pass
        
        surface = pygame.Surface(screen_size)
        surface.fill(color)
        surface = surface.convert()
        
        return surface
    
    def _create_default_terrain(self, screen_size: Tuple[int, int]) -> pygame.Surface:
        """创建默认地形"""
        surface = pygame.Surface(screen_size)
        surface.fill((34, 139, 34))  # 森林绿
        surface = surface.convert()
        return surface
    
    def batch_load_sprites(self, sprite_names: list, target_sizes: Dict[str, Tuple[int, int]] = None) -> Dict[str, pygame.Surface]:
        """批量加载精灵"""
        results = {}
        target_sizes = target_sizes or {}
        
        for name in sprite_names:
            size = target_sizes.get(name)
            results[name] = self.load_sprite(name, size)
        
        return results
    
    def preload_all_sprites(self):
        """预加载所有默认精灵"""
        sprite_names = list(self.default_sizes.keys())
        return self.batch_load_sprites(sprite_names)
    
    def clear_cache(self):
        """清空缓存"""
        self.sprite_cache.clear()
        self.metadata_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cached_sprites': len(self.sprite_cache),
            'cache_size_mb': sum(s.get_size()[0] * s.get_size()[1] * 4 for s in self.sprite_cache.values()) / (1024 * 1024),
            'stats': self.load_stats.copy()
        }
    
    def save_sprite_metadata(self, metadata_file: str = 'sprite_metadata.json'):
        """保存精灵元数据"""
        metadata = {
            'default_sizes': self.default_sizes,
            'default_colors': self.default_colors,
            'load_stats': self.load_stats
        }
        
        metadata_path = self.images_dir / metadata_file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_sprite_metadata(self, metadata_file: str = 'sprite_metadata.json'):
        """加载精灵元数据"""
        metadata_path = self.images_dir / metadata_file
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.default_sizes.update(metadata.get('default_sizes', {}))
                self.default_colors.update(metadata.get('default_colors', {}))
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Failed to load sprite metadata: {e}")
    
    def get(self, sprite_name: str) -> pygame.Surface:
        """获取精灵的简化接口，兼容原有代码"""
        return self.load_sprite(sprite_name)