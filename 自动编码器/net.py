import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self, input_size, hidden_size).__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.decode = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input_size, hidden_size):
        pass
