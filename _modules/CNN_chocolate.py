import torch.nn.functional as F
import torch.nn as nn
import params as p
import torch


class ConvNet(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=p.ch_out1, kernel_size=(3, p.k_w), stride=(1, p.s_w))
        self.bn1 = nn.BatchNorm2d(p.ch_out1, False)

        self.conv2 = nn.Conv2d(in_channels=p.ch_out1, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(1, False)
        
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 239))

    def forward(self, x):
        x = F.pad(x, (0, 0, 1, 1), mode='replicate')
        
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, p.dp_rate)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.dropout(x, p.dp_rate)

        x = self.avgpool(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2]))

        return x
