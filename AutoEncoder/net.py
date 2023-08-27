import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    基本自编码器
    """

    def __init__(self, input_size, hidden_size, LBL=False):
        """

        :param input_size:输入的尺寸
        :param hidden_size:中间隐层的尺寸
        :param LBL:是否是layer by layer 的训练
        """
        super(Autoencoder, self).__init__()
        self.LBL = LBL
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        if not self.LBL:

            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ConvAutoEncoder(nn.Module):
    """
    卷积自编码器
    """

    def __init__(self, input_channels=1, encoder_channels=[16, 32, 64], decoder_channels=[32, 16]):
        """

        :param input_channels:输入数据的通道数,default=1
        :param encoder_channels:需要一个列表,default=[16,32,64]
        :param decoder_channels:需要一个列表,default=[32,16]
        """
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels[0], kernel_size=3, stride=2, padding=1),
            # (1, 28, 28) -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=3, stride=2, padding=1),
            # (16, 14, 14) -> (32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(encoder_channels[1], encoder_channels[2], kernel_size=3),  # (32, 7, 7) -> (64, 5, 5)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[2], decoder_channels[0], kernel_size=3),  # (64, 5, 5) -> (32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=3, stride=2, padding=1,
                               output_padding=1),  # (32, 7, 7) -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_channels[1], input_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1),  # (16, 14, 14) -> (1, 28, 28)
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


class StackedAutoEncoder(nn.Module):
    """
    堆栈自动编码器 -- 非逐层训练策略
    """

    def __init__(self, input_size=784,
                 encoder_sizes=[128, 64, 32],
                 decoder_sizes=[64, 128]):
        """

        :param input_size:
        :param encoder_sizes:
        :param decoder_sizes:
        """
        super(StackedAutoEncoder, self).__init__()

        encoder_layers = []
        previous_size = input_size
        for size in encoder_sizes:
            encoder_layers.append(nn.Linear(previous_size, size))
            encoder_layers.append(nn.ReLU())
            previous_size = size
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        previous_size = encoder_sizes[-1]
        for size in decoder_sizes:
            decoder_layers.append(nn.Linear(previous_size, size))
            decoder_layers.append(nn.ReLU())
            previous_size = size
        decoder_layers.append(nn.Linear(previous_size, input_size))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class StackedAutoEncoderPro(nn.Module):
    """
    堆栈自动编码器 -- 逐层训练策略
    """

    def __init__(self, autoencoders):
        super(StackedAutoEncoderPro, self).__init__()

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


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
