import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F

from net import AutoencoderLinear, AutoencoderConv
import matplotlib.pyplot as plt
from data import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

train_dataloader = load_data()
line_model = AutoencoderLinear().to(device)
conv_model = AutoencoderConv().to(device)

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
    for data, _ in tqdm(train_dataloader, desc="batch"):
        data_con = data.to(device)
        data = data.view(-1, config["input_size"]).to(device)
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
