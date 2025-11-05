#!/usr/bin/env python3
"""
海战游戏启动器

用法:
  python run_game.py                  # 展示菜单
  python run_game.py --skip-menu      # 跳过菜单，使用默认地图
  python run_game.py --terrain map.png # 使用指定地形图
"""

import sys
import os

# 将当前目录添加到 Python 路径，确保能导入 src 包
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

"""
单一入口：统一从此脚本启动，并支持调试与菜单操作。
"""
if __name__ == '__main__':
    from src.core.game_single_play import Game
    from src.ui.start_menu import StartMenu
    from src.init import Scenario
    
    # 传递所有命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='海战游戏单人模式（单一入口）')
    parser.add_argument('--skip-menu', action='store_true', help='跳过启动菜单，直接进入游戏')
    parser.add_argument('--terrain', type=str, default=None, help='指定地形图片（src/render/map/）或纯色，如 color:#1E90FF 或 color:0,0,255')
    parser.add_argument('--auto-select-timeout', type=float, default=None, help='启动菜单自动选择/跳过的超时时间（秒）；默认禁用自动开始')
    parser.add_argument('--speed-factor', type=float, default=None, help='全局移动速度系数（例如 0.5 降速，1.0 正常）')
    parser.add_argument('--scenario', type=str, default=None, help='指定想定 JSON 文件（位于 src/core/data/）')
    parser.add_argument('--load-save', type=str, default=None, help='指定存档 JSON 文件（位于 saves/）')
    parser.add_argument('--save-on-exit', action='store_true', help='退出时自动保存当前状态到 saves/auto_last.json')
    parser.add_argument('--layered-rendering', action='store_true', help='启用分层渲染系统（优化性能和视觉效果）')
    parser.add_argument('--debug-rendering', action='store_true', help='启用渲染调试模式（显示性能统计和调试信息）')
    parser.add_argument('--grid-mode', type=str, choices=['square', 'hex', 'none'], default=None,
                        help='地图网格显示模式：square（默认）、hex（六角格）、none（不显示网格）')
    args = parser.parse_args()

    # 定义全局配置字典
    game_config = {
        'name': 'AirDefense',
        'device_path': 'core/data/device_new.json',
        'scenario_path': 'core/data/skirmish_1.json',
        'map_path': 'core/data/map.json',
        'movement_speed_factor': 0.5,
        'grid_mode': 'square',
    }

    players = {
        "red": 'Red',
        "blue": 'Blue'
    }

    # 如果指定了速度因子，更新配置
    if args.speed_factor is not None:
        game_config['movement_speed_factor'] = float(args.speed_factor)
    
    # 添加渲染配置
    game_config['use_layered_rendering'] = args.layered_rendering
    game_config['debug_rendering'] = args.debug_rendering
    if args.grid_mode:
        game_config['grid_mode'] = args.grid_mode
    
    game = Game(game_config, players)

    selected_map = None
    selected_action = None
    selected_scenario = None
    selected_save = None
    if not args.skip_menu and not (args.terrain or args.scenario or args.load_save):
        # 展示扩展菜单，返回结构化选择
        menu = StartMenu()
        choice = menu.run_extended(screen_size=(1280, 800), auto_select_timeout=args.auto_select_timeout)
        if choice is None:
            selected_action = 'exit'
        else:
            selected_action = choice.get('action')
            selected_map = choice.get('terrain')
            selected_scenario = choice.get('scenario')
            selected_save = choice.get('save')

    # CLI参数优先级：显式传入覆盖菜单选择
    if args.terrain:
        selected_action = 'start'
        selected_map = args.terrain
    if args.scenario:
        selected_action = 'open_scenario'
        selected_scenario = args.scenario
    if args.load_save:
        selected_action = 'load_save'
        selected_save = args.load_save

    try:
        # 根据选择执行对应操作
        if selected_action == 'open_scenario' and selected_scenario:
            # 更新想定路径并重置
            game.env.scenario = Scenario(os.path.join('core', 'data', selected_scenario))
            game.env.reset_game()
            print(f'已加载想定: {selected_scenario}，请点击开始进入或使用 --terrain 指定地图。')
            # 如果同时选择了地图，直接启动
            if selected_map:
                print(f'正在启动游戏... 地形: {selected_map}')
                game.run(terrain_override=selected_map)
            else:
                # 尝试使用场景绑定的地图图片
                bound = getattr(game.env, 'default_map_image', None)
                if bound:
                    print(f'检测到场景绑定地图: {bound}，自动启动。')
                    game.run(terrain_override=bound)
                else:
                    # 回到菜单或等待用户以 CLI 传地图
                    print('未选择地图，保持菜单/退出。')
        elif selected_action == 'load_save' and selected_save:
            save_path = os.path.join(current_dir, 'saves', selected_save)
            # 在读取存档前确保已有实体
            game.env.reset_game()
            if game.env.load_state(save_path):
                print(f'已读取存档: {selected_save}，开始游戏。')
                game.run(terrain_override=selected_map)
            else:
                print('读取存档失败。')
        elif selected_action == 'start' and (selected_map or args.terrain):
            print(f'正在启动游戏... 地形: {selected_map or args.terrain or "默认"}')
            game.run(terrain_override=selected_map or args.terrain)
        else:
            print('未选择有效操作，已退出或返回菜单。')
    except KeyboardInterrupt:
        print('用户中断，正在退出...')
    finally:
        # 优雅停止通信线程
        if hasattr(game, 'communication_client'):
            try:
                game.communication_client.stop()
            except Exception:
                pass
        if hasattr(game, 'communication_server'):
            try:
                game.communication_server.stop()
            except Exception:
                pass
        # 自动保存
        if args.save_on_exit:
            auto_save = os.path.join(current_dir, 'saves', 'auto_last.json')
            try:
                os.makedirs(os.path.dirname(auto_save), exist_ok=True)
                game.env.save_state(auto_save)
                print(f'已自动保存到: {auto_save}')
            except Exception as e:
                print(f'自动保存失败: {e}')
        print('已退出')