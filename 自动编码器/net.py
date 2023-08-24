import torch.nn as nn


class AutoencoderLinear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoencoderLinear, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoencoderConv(nn.Module):
    def __init__(self):
        super(AutoencoderConv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (1, 28, 28) -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16, 14, 14) -> (32, 7, 7)
            nn.ReLU(),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (32, 7, 7) -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (16, 14, 14) -> (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
