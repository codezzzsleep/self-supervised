from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net import AutoencoderLinear, AutoencoderConv
import matplotlib.pyplot as plt

config = {
    "input_size": 784,
    "hidden_size": 121,
    "output_size": 10,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "weight_decay": 1e-5,
    "model_select": "all"
}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

line_model = AutoencoderLinear(config["input_size"], config["hidden_size"])
conv_model = AutoencoderConv()
train_dataset = datasets.MNIST(root="./data",
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(train_dataset,
                          batch_size=config["batch_size"],
                          shuffle=True)

optimizer = Adam(line_model.parameters(),
                 lr=config["learning_rate"],
                 weight_decay=config["weight_decay"])
optimizer_conv = Adam(conv_model.parameters(),
                      lr=config["learning_rate"],
                      weight_decay=config["weight_decay"])
line_model.train()
conv_model.train()
epochs = config["epochs"]
auto_line_loss = []
auto_cov_loss = []
x = list(range(1, epochs + 1))

for epoch in tqdm(range(epochs), desc="epoch"):
    for data, _ in tqdm(train_loader, desc="batch"):
        data_con = data
        data = data.view(-1, config["input_size"])
        optimizer.zero_grad()
        optimizer_conv.zero_grad()
        output_line = line_model(data)
        output_conv = conv_model(data_con)
        loss_line = F.mse_loss(output_line, data)
        loss_conv = F.mse_loss(output_conv, data_con)
        loss_line.backward()
        loss_conv.backward()
        optimizer.step()
        optimizer_conv.step()
    print()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_line.item():.4f}")
    auto_line_loss.append(loss_line.item())
    auto_cov_loss.append(loss_conv.item())

print("done!")

epochs = list(range(1, 101))

plt.plot(epochs, auto_line_loss, color='blue', label='auto_line_loss')
plt.plot(epochs, auto_cov_loss, color='red', label='auto_cov_loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.title('Loss vs Epoch for Different Models')

plt.show()
