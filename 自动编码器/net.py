import torch
import torch.nn as nn


class LinearAutoencoder(nn.Module):
    """use linear encoder and decoder"""

    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ConvAutoencoder(nn.Module):
    """use Conv encoder and decoder"""

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (1, 28, 28) -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=3, padding=1),  # (16, 14, 14) -> (32, 5, 5)
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=3, padding=1, output_padding=1),  # (32, 5, 5) -> (16, 15, 15)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (16, 15, 15) -> (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class StackedAutoEncoder(nn.Module):
    """堆栈自动编码器--没有使用逐层训练策略"""

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        # Encoding layers
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # (784, 128)
            nn.ReLU(True),
            nn.Linear(128, 64),  # (128, 64)
            nn.ReLU(True),
            nn.Linear(64, 32),  # (64, 32)
            nn.ReLU(True)
        )

        # Decoding layers
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),  # (32, 64)
            nn.ReLU(True),
            nn.Linear(64, 128),  # (64, 128)
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),  # (128, 784)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    """单层的模型，适应逐层训练策略"""

    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class StackedAutoEncoder(nn.Module):
    """不参与训练，在单层训练完成后，集成使用"""
    def __init__(self, autoencoders):
        super(StackedAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            autoencoders[0].encoder,
            autoencoders[1].encoder,
            autoencoders[2].encoder
        )

        self.decoder = nn.Sequential(
            autoencoders[2].decoder,
            autoencoders[1].decoder,
            autoencoders[0].decoder
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MyLeNet5(nn.Module):
    """修改后的 LeNet5 网络--对28*28的数据做了适应"""

    def __init__(self, num_classes=10):
        super(MyLeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
