import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, fc_layer_size, dropout):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Linear(64 * 7 * 7, fc_layer_size, bias=True), nn.ReLU(),
            nn.Dropout2d(p=dropout))
        self.layer4 = nn.Sequential(
            nn.Linear(fc_layer_size, 84), nn.ReLU(),
            nn.Dropout2d(p=dropout))
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1) 
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc3(x)
        return x

