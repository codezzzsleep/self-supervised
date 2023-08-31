import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x))


# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        return self.activate(self.fc2(x))


# Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rho=0.05, beta=3.):
        super(SparseAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.rho = rho
        self.beta = beta

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs


# KL Divergence Loss
def kl_divergence_loss(encoder_output, rho, batch_size):
    rho_hat = torch.mean(encoder_output, dim=0) + 1e-8  # 在rho_hat上添加一个小数值,防止除0
    kl_loss = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
    return kl_loss / batch_size


# Training parameters
input_size = 784
hidden_size = 128
output_size = 784
epochs = 50
batch_size = 64
learning_rate = 0.0001
# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("runs/sp_log")
# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Creating the model and setting up optimizer and criterion
sparse_ae = SparseAutoencoder(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(sparse_ae.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
rho = 0.05
beta = 3.

for epoch in range(epochs):
    running_loss = 0.
    for i, data in enumerate(dataloader):
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1).to(device)

        optimizer.zero_grad()

        # Forward pass
        decoder_outputs = sparse_ae(inputs)
        reconstruction_loss = criterion(decoder_outputs, inputs)
        running_loss += reconstruction_loss.item()
        encoder_outputs = sparse_ae.encoder(inputs)
        kl_loss = kl_divergence_loss(encoder_outputs, rho, inputs.size(0))

        # Total loss
        total_loss = reconstruction_loss + beta * kl_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()
    writer.add_scalar("Loss/train", running_loss / (i + 1), epoch)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")
