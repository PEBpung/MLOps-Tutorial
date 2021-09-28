import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, fc_layer_size, dropout):
        # We optimize dropout rate in a convolutional neural network.
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        self.drop1=nn.Dropout2d(p=dropout)   
        self.fc1 = nn.Linear(32 * 7 * 7, fc_layer_size)
        self.drop2=nn.Dropout2d(p=dropout)
        self.fc2 = nn.Linear(fc_layer_size, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2))
        x = self.drop1(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

