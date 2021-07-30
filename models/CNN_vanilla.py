from utils import params as p

import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvNet(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, p.k_w), stride=(1, p.s_w))
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 239)) 
        self.soft = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        # print("input: ",x.shape)
        x = F.pad(x, (0, 0, 1, 1), mode='replicate')
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2]))
        
        return x
