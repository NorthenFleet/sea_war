from player_base import Player_Base


class Player_AI(Player_Base):
    def __init__(self, name):
        super().__init__(name)
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        raise NotImplementedError(
            "This method should be overridden by subclasses")
