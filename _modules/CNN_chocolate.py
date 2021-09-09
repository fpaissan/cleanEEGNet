import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvNet(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=8,
                              kernel_size=(3, 256),
                              stride=(1, 128))
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=32,
                               kernel_size=(5, 5),
                               stride=(2, 2),
                               groups=8)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               groups=32)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.avgpool = nn.AvgPool2d(kernel_size=(25, 4))

        self.lin = nn.Linear(64, 62)
    
    def forward(self, x):
        bs = x.shape[0]
        x = F.pad(x, (0, 0, 1, 1), mode='replicate')
        
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.avgpool(x).view(bs, 64)

        x = self.lin(x)

        return x
