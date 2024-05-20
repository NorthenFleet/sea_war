from player_base import Player_Base
import torch
from model_select import *


class AIPlayer(Player_Base):
    def __init__(self, AI_config):
        super(AIPlayer, self).__init__()
        self.state_size = AI_config["state_size"]
        self.action_size = AI_config["action_size"]
        config = {
            "model_type": AI_config["model"],
            "input_dim": self.state_size,
            "output_dim": self.action_size
        }
    
        self.modle = model_select(**config)

        self.agents = {}
        # for name, (module, cls, training_config) in AI_config.items():
        #     agent_class = getattr(__import__(module), cls)
        #     if training_config["model"] is not None:
        #         self.players[name] = agent_class(name, training_config)
        #     else:
        #         self.players[name] = agent_class(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    def save_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode


if __name__ == '__main__':
    AIPlayer.save_model("ppo", "000")
    print("保存网络")
