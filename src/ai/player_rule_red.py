from ui.player import Player
from core.game_data import GameData


class RedPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

        self.units = None

    def choose_action(self, side):
        self.units = side
        print("我是红方智能体")
