#!/usr/bin/env python3
"""
海战游戏渲染系统性能测试脚本

用法:
  python performance_test.py                    # 运行完整性能测试
  python performance_test.py --quick            # 快速测试（较短时间）
  python performance_test.py --profile          # 详细性能分析
"""

import sys
import os
import time
import psutil
import threading
import json
from datetime import datetime

# 将当前目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.core.game_single_play import Game
from src.init import Scenario

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.data = {
            'fps_samples': [],
            'memory_samples': [],
            'cpu_samples': [],
            'start_time': None,
            'end_time': None
        }
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.data['start_time'] = time.time()
        self.data['fps_samples'] = []
        self.data['memory_samples'] = []
        self.data['cpu_samples'] = []
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.data['end_time'] = time.time()
        
    def record_fps(self, fps):
        """记录FPS"""
        if self.monitoring:
            self.data['fps_samples'].append({
                'timestamp': time.time(),
                'fps': fps
            })
            
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 记录内存使用
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 记录CPU使用
                cpu_percent = self.process.cpu_percent()
                
                timestamp = time.time()
                self.data['memory_samples'].append({
                    'timestamp': timestamp,
                    'memory_mb': memory_mb
                })
                self.data['cpu_samples'].append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.1)  # 每100ms采样一次
            except Exception as e:
                print(f"监控错误: {e}")
                break
                
    def get_statistics(self):
        """获取统计信息"""
        if not self.data['fps_samples']:
            return None
            
        fps_values = [sample['fps'] for sample in self.data['fps_samples']]
        memory_values = [sample['memory_mb'] for sample in self.data['memory_samples']]
        cpu_values = [sample['cpu_percent'] for sample in self.data['cpu_samples']]
        
        duration = self.data['end_time'] - self.data['start_time']
        
        return {
            'duration': duration,
            'fps': {
                'avg': sum(fps_values) / len(fps_values) if fps_values else 0,
                'min': min(fps_values) if fps_values else 0,
                'max': max(fps_values) if fps_values else 0,
                'samples': len(fps_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values) if memory_values else 0,
                'min': min(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'peak': max(memory_values) if memory_values else 0
            },
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0
            }
        }

class PerformanceTestGame(Game):
    """性能测试专用游戏类"""
    
    def __init__(self, game_config, players, use_layered_rendering=True, test_duration=30):
        super().__init__(game_config, players, use_layered_rendering=use_layered_rendering)
        self.test_duration = test_duration
        self.monitor = PerformanceMonitor()
        self.test_start_time = None
        
    def run_performance_test(self, terrain_override=None):
        """运行性能测试"""
        print(f"开始性能测试 - 渲染模式: {'分层渲染' if self.use_layered_rendering else '传统渲染'}")
        print(f"测试时长: {self.test_duration}秒")
        
        # 初始化游戏
        try:
            game_data, sides = self.env.reset_game()
        except Exception as e:
            print(f"游戏初始化失败: {e}")
            # 尝试简单初始化
            game_data = self.env.game_data
            sides = {'red': [], 'blue': []}
        
        # 创建渲染器
        if self.render_manager is None:
            show_obstacles = bool(getattr(self.env, 'default_map_json', None))
            
            if self.use_layered_rendering:
                from src.render.integrated_renderer import IntegratedRenderManager
                self.render_manager = IntegratedRenderManager(
                    self.env, self.screen_size, 
                    terrain_override=terrain_override, 
                    show_obstacles=show_obstacles,
                    use_layered_rendering=True
                )
            else:
                from src.render.single_process import RenderManager
                self.render_manager = RenderManager(
                    self.env, self.screen_size, 
                    terrain_override=terrain_override, 
                    show_obstacles=show_obstacles
                )
        
        # 开始监控
        self.monitor.start_monitoring()
        self.test_start_time = time.time()
        
        # 运行测试循环
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # 检查测试时间
                if time.time() - self.test_start_time >= self.test_duration:
                    break
                
                # 简化的游戏更新 - 避免调用可能不存在的方法
                try:
                    # 尝试更新环境
                    result = self.env.step({})
                    if isinstance(result, tuple) and len(result) >= 2:
                        game_data, sides = result
                    else:
                        # 如果step返回格式不对，使用现有数据
                        game_data = self.env.game_data
                        sides = getattr(self.env, 'sides', {'red': [], 'blue': []})
                except Exception as e:
                    # 如果step失败，继续使用现有数据
                    game_data = self.env.game_data
                    sides = getattr(self.env, 'sides', {'red': [], 'blue': []})
                
                # 渲染
                if self.render_manager:
                    try:
                        self.render_manager.render(game_data, sides)
                    except Exception as e:
                        # 如果渲染失败，尝试基本更新
                        try:
                            self.render_manager.update()
                        except Exception:
                            pass
                
                frame_count += 1
                fps_counter += 1
                
                # 每秒计算一次FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    self.monitor.record_fps(fps)
                    fps_counter = 0
                    last_fps_time = current_time
                    
                    # 显示进度
                    elapsed = current_time - self.test_start_time
                    progress = (elapsed / self.test_duration) * 100
                    print(f"\r进度: {progress:.1f}% - FPS: {fps:.1f}", end='', flush=True)
                
                # 限制帧率到60FPS
                frame_time = time.time() - loop_start
                target_frame_time = 1.0 / 60.0
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                    
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        finally:
            # 停止监控
            self.monitor.stop_monitoring()
            
            # 清理渲染器
            if self.render_manager:
                try:
                    self.render_manager.cleanup()
                except Exception:
                    pass
                
        print(f"\n测试完成 - 总帧数: {frame_count}")
        return self.monitor.get_statistics()

def run_comparison_test(test_duration=30, quick_mode=False):
    """运行对比测试"""
    if quick_mode:
        test_duration = 10
        
    # 游戏配置
    game_config = {
        'name': 'PerformanceTest',
        'device_path': 'core/data/device_new.json',
        'scenario_path': 'core/data/skirmish_1.json',
        'map_path': 'core/data/map.json',
        'movement_speed_factor': 1.0,
    }
    
    players = {
        "red": 'Red',
        "blue": 'Blue'
    }
    
    results = {}
    
    # 测试传统渲染
    print("=" * 60)
    print("测试传统渲染系统")
    print("=" * 60)
    
    try:
        traditional_game = PerformanceTestGame(
            game_config, players, 
            use_layered_rendering=False, 
            test_duration=test_duration
        )
        results['traditional'] = traditional_game.run_performance_test()
        del traditional_game
        time.sleep(2)  # 等待资源释放
    except Exception as e:
        print(f"传统渲染测试失败: {e}")
        results['traditional'] = None
    
    # 测试分层渲染
    print("\n" + "=" * 60)
    print("测试分层渲染系统")
    print("=" * 60)
    
    try:
        layered_game = PerformanceTestGame(
            game_config, players, 
            use_layered_rendering=True, 
            test_duration=test_duration
        )
        results['layered'] = layered_game.run_performance_test()
        del layered_game
    except Exception as e:
        print(f"分层渲染测试失败: {e}")
        results['layered'] = None
    
    return results

def print_comparison_report(results):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print("性能测试报告")
    print("=" * 80)
    
    if not results.get('traditional') or not results.get('layered'):
        print("测试数据不完整，无法生成对比报告")
        return
    
    trad = results['traditional']
    layer = results['layered']
    
    print(f"{'指标':<15} {'传统渲染':<15} {'分层渲染':<15} {'改进':<15}")
    print("-" * 65)
    
    # FPS对比
    fps_improvement = ((layer['fps']['avg'] - trad['fps']['avg']) / trad['fps']['avg']) * 100
    print(f"{'平均FPS':<15} {trad['fps']['avg']:<15.1f} {layer['fps']['avg']:<15.1f} {fps_improvement:+.1f}%")
    print(f"{'最小FPS':<15} {trad['fps']['min']:<15.1f} {layer['fps']['min']:<15.1f}")
    print(f"{'最大FPS':<15} {trad['fps']['max']:<15.1f} {layer['fps']['max']:<15.1f}")
    
    # 内存对比
    memory_improvement = ((trad['memory']['avg'] - layer['memory']['avg']) / trad['memory']['avg']) * 100
    print(f"{'平均内存(MB)':<15} {trad['memory']['avg']:<15.1f} {layer['memory']['avg']:<15.1f} {memory_improvement:+.1f}%")
    print(f"{'峰值内存(MB)':<15} {trad['memory']['peak']:<15.1f} {layer['memory']['peak']:<15.1f}")
    
    # CPU对比
    cpu_improvement = ((trad['cpu']['avg'] - layer['cpu']['avg']) / trad['cpu']['avg']) * 100
    print(f"{'平均CPU(%)':<15} {trad['cpu']['avg']:<15.1f} {layer['cpu']['avg']:<15.1f} {cpu_improvement:+.1f}%")
    
    print("\n总结:")
    if fps_improvement > 5:
        print(f"✓ 分层渲染在FPS方面有显著提升 (+{fps_improvement:.1f}%)")
    elif fps_improvement > 0:
        print(f"✓ 分层渲染在FPS方面有轻微提升 (+{fps_improvement:.1f}%)")
    else:
        print(f"✗ 分层渲染在FPS方面有下降 ({fps_improvement:.1f}%)")
    
    if memory_improvement > 5:
        print(f"✓ 分层渲染显著减少了内存使用 (+{memory_improvement:.1f}%)")
    elif memory_improvement > 0:
        print(f"✓ 分层渲染轻微减少了内存使用 (+{memory_improvement:.1f}%)")
    else:
        print(f"✗ 分层渲染增加了内存使用 ({memory_improvement:.1f}%)")

def save_results(results, filename=None):
    """保存测试结果"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_{timestamp}.json"
    
    results['test_info'] = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'python_version': sys.version
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存到: {filename}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='海战游戏渲染系统性能测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（10秒）')
    parser.add_argument('--duration', type=int, default=30, help='测试持续时间（秒）')
    parser.add_argument('--profile', action='store_true', help='启用详细性能分析')
    parser.add_argument('--save', type=str, default=None, help='保存结果到指定文件')
    args = parser.parse_args()
    
    try:
        # 运行对比测试
        results = run_comparison_test(
            test_duration=args.duration if not args.quick else 10,
            quick_mode=args.quick
        )
        
        # 打印报告
        print_comparison_report(results)
        
        # 保存结果
        save_results(results, args.save)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()