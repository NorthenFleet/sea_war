import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode
