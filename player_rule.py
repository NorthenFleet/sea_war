from player import Player
from game_data import GameData


class RulePlayer(Player):
    def __init__(self, name, game_data):
        super().__init__()
        self.name = name
        self.game_data = game_data
        self.units = None

    def choose_action(self, state):
        self.units = self.game_data.get_all_unit_ids()
        print("我是规则智能体")
