from ..ui.player import Player, CommandList, Command, MoveCommand
from ..core.entities.entity import PositionComponent, MovementComponent
from ..core.game_data import GameData
import numpy as np


class BluePlayer(Player):
    def __init__(self, name, device_table):
        super().__init__(name)
        self.name = name
        self.device_table = device_table

    def choose_action(self, side):
        # 基于规则：每个蓝方单位向最近的敌方单位移动
        command_list = CommandList()
        Command.set_command_list(command_list)

        gd = GameData._instance
        if gd is None:
            return command_list

        enemies = [e for e in gd.get_all_entities()
                   if gd.get_unit_owner(e.entity_id) != side.name]

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
                d = np.linalg.norm(epos.get_param('position')[:2] - my_pos[:2])
                if d < min_dist:
                    min_dist = d
                    nearest = epos.get_param('position')

            if nearest is None:
                continue

            mv = entity.get_component(MovementComponent)
            speed = mv.get_param('speed') if mv else 1
            MoveCommand(entity.entity_id, target_position=(int(nearest[0]), int(nearest[1])), speed=speed)

        return command_list
