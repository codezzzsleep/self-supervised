import torch.nn.functional as F
from net import LinearAutoencoder
from torch.optim import Adam
from tqdm import tqdm

model = LinearAutoencoder()
optimizer = Adam(model.parameters(), lr=0.01)

loss_list = []


def train(dataloader, epochs):
    # 请确保 数据和模型是在 传入训练之前送入正确的设备中
    for epoch in tqdm(range(epochs), desc="epoch"):
        for data, _ in dataloader:
            optimizer.zero_grad()
            data = data.view(-1, 784)
            output = model(data)
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
    print("line done!")
