from utils import params as p

import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvNet3(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 2500), stride=(1, 1250))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.rnn = nn.RNN(input_size=1, hidden_size=1, batch_first=True)

    def forward(self, x):
        # print("input: ",x.shape)
        x = F.pad(x, (0, 0, 1, 1), mode='replicate')
        # print("padding: ", x.shape)
        x = F.relu(self.conv(x))
        # print("conv: ",x.shape)
        x = F.relu(self.conv2(x))
        hidden_state = torch.rand(x.shape[1], x.shape[0], 1).to(p.device)

        output = torch.zeros(x.shape[0], x.shape[2]).to(p.device)

        for i in range(x.shape[2]):
            inp = x[:, :, i, :]
            # input = torch.reshape(input,(input.shape[0],input.shape[1],input.shape[3]))
            inp = inp.transpose(1, 2)
            temp, _ = self.rnn(inp, hidden_state)
            # print(temp)
            # print(temp[:,-1,:].shape," ", output[:,i].shape)
            output[:, i] = temp[:, -1, 0]

        # print("maxpool: ",x.shape)
        # output = torch.reshape(output,(output.shape[0],x.shape[2]))
        output = torch.sigmoid(output)
        # print(output)
        return output
