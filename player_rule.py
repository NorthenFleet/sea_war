from player import Player
from game_data import GameData


class RulePlayer(Player):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.game_data = GameData()
        self.units = None

    def choose_action(self, state):
        self.units = self.game_data.get_all_units()
        print("我是规则智能体")
