import torch.nn as nn
import torch.optim as optim

# Conv2D equation
# output_size = (input_size - kernel_size + 2 * padding) / stride + 1

# MaxPool2D equation
# output_size = (input_size - kernel_size) / stride + 1

# CNN architecture
class MnistCNN(nn.Module):
    def __init__(self, input_size = 28, output_size = 10):
        super(MnistCNN, self).__init__()
        self.layer_01 = nn.Sequential(
            # Conv2D: (28 - 3 + 2 * 2) / 1 + 1 = 30
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 2),
            nn.ReLU(),
            # MaxPool2D: (30 - 2) / 2 + 1 = 15
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer_02 = nn.Sequential(
            # Conv2D: (15 - 5 + 2 * 2) / 1 + 1 = 15
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding = 2),
            nn.ReLU(),
            # MaxPool2D: (15 - 2) / 2 + 1 = 7
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer_out = nn.Linear(7 * 7 * 32, output_size)

    def forward(self, x):
        x = self.layer_01(x)
        x = self.layer_02(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer_out(x)

        return x