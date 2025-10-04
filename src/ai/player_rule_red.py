from ..ui.player import Player, CommandList, Command, MoveCommand
from ..core.entities.entity import PositionComponent, MovementComponent
from ..core.game_data import GameData
import numpy as np


class RedPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def choose_action(self, side):
        # 规则：曲线接近最近敌人，并在终点加入侧翼偏移
        command_list = CommandList()
        Command.set_command_list(command_list)

        gd = GameData._instance
        if gd is None:
            return command_list

        # 敌方实体集合
        enemies = [e for e in gd.get_all_entities()
                   if gd.get_unit_owner(e.entity_id) != side.name]

        flank_offset = 120.0  # 侧翼偏移
        curve_ratio = 0.35    # 曲线中间点偏移比例

        for entity in side.entities:
            pos_comp = entity.get_component(PositionComponent)
            if pos_comp is None:
                continue
            my_pos = pos_comp.get_param('position')

            # 找最近敌人
            nearest = None
            min_dist = float('inf')
            for enemy in enemies:
                epos = enemy.get_component(PositionComponent)
                if epos is None:
                    continue
                e_xy = np.array(epos.get_param('position')[:2], dtype=np.float64)
                my_xy = np.array(my_pos[:2], dtype=np.float64)
                d = float(np.linalg.norm(e_xy - my_xy))
                if d < min_dist:
                    min_dist = d
                    nearest = epos.get_param('position')

            if nearest is None:
                continue

            my_xy = np.array(my_pos[:2], dtype=np.float64)
            enemy_xy = np.array(nearest[:2], dtype=np.float64)
            delta = enemy_xy - my_xy
            d = float(np.linalg.norm(delta))
            if d < 1e-6:
                continue
            dir = delta / d

            # 垂直方向（侧翼）
            perp = np.array([-dir[1], dir[0]], dtype=np.float64)
            target_xy = enemy_xy + perp * flank_offset

            # 使用偏移终点作为路径目标（曲线由寻路系统处理）
            mv = entity.get_component(MovementComponent)
            speed = mv.get_param('speed') if mv else 1
            MoveCommand(entity.entity_id,
                        target_position=(int(target_xy[0]), int(target_xy[1])),
                        speed=speed)

        return command_list
