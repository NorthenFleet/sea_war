from player import Player


class RulePlayer(Player):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def choose_action(self, state):
        print("我是规则智能体")
