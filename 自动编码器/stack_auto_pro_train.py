import torch.nn.functional as F
from net import AutoEncoder
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

model = AutoEncoder()
# optimizer = Adam(model.parameters(), lr=0.01)

loss_list = []

model.train()


def train(dataloader, epochs, device, writer):
    model.to(device)
    autoencoders = [
        AutoEncoder(784, 128).to(device),
        AutoEncoder(128, 64).to(device),
        AutoEncoder(64, 32).to(device)
    ]
    for i, autoencoder in enumerate(autoencoders):
        print("Training Autoencoder Layer", i + 1)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(10):
            autoencoder.train()
            epoch_loss = 0
            for data, _ in train_loader:
                data = data.view(data.size(0), -1).to(device)
                optimizer.zero_grad()
                output = autoencoder(data)
                loss = F.mse_loss(output, data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print("Epoch [{}]: Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))

        if i < len(autoencoders) - 1:
            with torch.no_grad():
                new_data = []
                for data, _ in train_loader:
                    data = data.view(data.size(0), -1).to(device)
                    features = autoencoder.encoder(data)
                    new_data.append(features)
                new_dataset = torch.cat(new_data)
                train_loader = DataLoader(new_dataset, batch_size=256, shuffle=True)


def one_train(dataloader, device):
    model.to(device)
    run_loss = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, momentum=0.9)
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = data
        data = x.view(-1, 784)
        data = data.to(device)
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    return run_loss / (i + 1)
