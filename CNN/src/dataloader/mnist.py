import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset

# Define custom transform
custom_transform = transforms.Compose([
    transforms.ToTensor(),
])

class MNISTDataset(Dataset):
    def __init__(self, data, target, transform = custom_transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        x = self.transform(x)

        return x, y