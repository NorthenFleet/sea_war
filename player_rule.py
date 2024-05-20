from player_base import Player_Base


class RulePlayer(Player_Base):
    def __init__(self):
        super().__init__()
        
    def choose_action(self, state):
        print("我是规则智能体")