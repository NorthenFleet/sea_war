from ..ui.player import Player, CommandList, Command, MoveCommand, StopCommand
from ..core.entities.entity import PositionComponent, MovementComponent
from ..core.game_data import GameData
import numpy as np


class BluePlayer(Player):
    def __init__(self, name, device_table):
        super().__init__(name)
        self.name = name
        self.device_table = device_table

    def choose_action(self, side):
        # 规则：保持安全距离接近最近敌人，过近则停止
        command_list = CommandList()
        Command.set_command_list(command_list)

        gd = GameData._instance
        if gd is None:
            return command_list

        enemies = [e for e in gd.get_all_entities()
                   if gd.get_unit_owner(e.entity_id) != side.name]

        safe_distance = 200.0  # 安全距离（单位坐标）
        approach_margin = 10.0  # 到达判据外扩，避免抖动

        for entity in side.entities:
            pos_comp = entity.get_component(PositionComponent)
            if pos_comp is None:
                continue
            my_pos = pos_comp.get_param('position')

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

            # 计算与敌人的距离与方向
            my_xy = np.array(my_pos[:2], dtype=np.float64)
            enemy_xy = np.array(nearest[:2], dtype=np.float64)
            vec = my_xy - enemy_xy
            dist = float(np.linalg.norm(vec))
            if dist <= (safe_distance - approach_margin):
                # 已过近：停止，避免贴脸
                StopCommand(entity.entity_id)
                continue

            # 目标为敌人外侧安全圈上的一点（沿敌->我方向）
            dir_vec = vec / (dist if dist > 1e-6 else 1.0)
            target_xy = enemy_xy + dir_vec * safe_distance

            mv = entity.get_component(MovementComponent)
            speed = mv.get_param('speed') if mv else 1
            MoveCommand(entity.entity_id,
                        target_position=(int(target_xy[0]), int(target_xy[1])),
                        speed=speed)

        return command_list
