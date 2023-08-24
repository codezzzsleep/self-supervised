import torch.nn as nn


class AutoencoderLinear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoencoderLinear, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activate = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x
