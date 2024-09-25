from player import Player
from game_data import GameData


class RulePlayer(Player):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.units = None

    def choose_action(self, game_data):
        self.units = game_data.get_all_unit_ids()
        print("我是规则智能体")
