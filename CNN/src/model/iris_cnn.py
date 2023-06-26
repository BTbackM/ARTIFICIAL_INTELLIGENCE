import torch.nn as nn
import torch.optim as optim

# CNN architecture
class IrisCNN(nn.Module):
    def __init__(self, input_size = 4, output_size = 3):
        super(IrisCNN, self).__init__()
        self.layer_01 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
        )
        self.layer_02 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.layer_03 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.layer_out = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.layer_01(x)
        x = self.layer_02(x)
        x = self.layer_03(x)
        x = self.layer_out(x)

        return x