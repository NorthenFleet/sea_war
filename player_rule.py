from player import Player
from game_data import GameData


class RulePlayer(Player):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.game_data = GameData()

    def choose_action(self, state):
        self.game_data
        print("我是规则智能体")
