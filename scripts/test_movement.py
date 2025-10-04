import os
import sys
import time
import math

# 将 src 加入模块路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.core.sea_war_env import SeaWarEnv
from src.core.entities.entity import PositionComponent, MovementComponent


def main():
    game_config = {
        'name': 'AirDefense',
        'device_path': 'core/data/device_new.json',
        'scenario_path': 'core/data/skirmish_1.json',
        'map_path': 'core/data/map.json',
    }

    env = SeaWarEnv(game_config)
    game_data, sides = env.reset_game()

    # 选择若干可移动单位进行跟踪（speed > 0）
    tracked = []
    for e in game_data.get_all_entities():
        mv = e.get_component(MovementComponent)
        pos = e.get_component(PositionComponent)
        if mv and pos:
            spd = mv.get_param('speed')
            if spd and spd > 0:
                # 保存初始位置拷贝
                p = pos.get_param('position')
                tracked.append((e.entity_id, [float(p[0]), float(p[1]), float(p[2])]))
        if len(tracked) >= 5:
            break

    if not tracked:
        print('No movable entities found. Scenario generation may have only static units.')
        return 0

    print('Tracked entities (initial positions):')
    for eid, p in tracked:
        print(f"  {eid}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    # 运行 120 帧（约 2 秒），验证位移
    steps = 120
    dt = 1.0 / 60.0
    for _ in range(steps):
        env.update(dt)

    print('Tracked entities (after 120 steps):')
    moved = 0
    moved = 0
    for eid, p0 in tracked:
        e = game_data.get_entity(eid)
        pos = e.get_component(PositionComponent).get_param('position')
        print(f"  {eid}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        # 检查相对初始位置是否位移
        dx = float(pos[0]) - float(p0[0])
        dy = float(pos[1]) - float(p0[1])
        if abs(dx) + abs(dy) > 1e-3:
            moved += 1

    print(f"Movable tracked count: {moved} / {len(tracked)}")
    # 若至少一个可移动单位存在，认为移动系统工作（详细数值由渲染验证）
    return 0 if moved > 0 else 1


if __name__ == '__main__':
    code = main()
    sys.exit(code)