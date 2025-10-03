from ..ui.player import Player
from ..core.game_data import GameData


class RulePlayer(Player):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.units = None

    def choose_action(self, side):
        self.units = side
        print("我是规则智能体")

        cmd = []

        return cmd
