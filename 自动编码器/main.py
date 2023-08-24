from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net import AutoencoderLinear

config = {
    "input_size": 784,
    "hidden_size": 128,
    "output_size": 10,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "weight_decay": 1e-5
}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

model = AutoencoderLinear(config["input_size"], config["hidden_size"])

train_dataset = datasets.MNIST(root="./data",
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(train_dataset,
                          batch_size=config["batch_size"],
                          shuffle=True)

optimizer = Adam(model.parameters(),
                 lr=config["learning_rate"],
                 weight_decay=config["weight_decay"])
model.train()
epochs = config["epochs"]

for epoch in tqdm(range(epochs), desc="epoch"):
    for data, _ in tqdm(train_loader, desc="batch"):
        data = data.view(-1, config["input_size"])
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("done!")
