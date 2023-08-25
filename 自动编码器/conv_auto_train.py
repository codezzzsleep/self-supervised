import torch.nn.functional as F
from net import ConvAutoencoder
from torch.optim import Adam
from tqdm import tqdm

model = ConvAutoencoder()
optimizer = Adam(model.parameters(), lr=0.01)

loss_list = []
model.train()


def train(dataloader, epochs, device, writer):
    model.to(device)
    for epoch in tqdm(range(epochs), desc="epoch"):
        run_loss = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = data
            data = x.to(device)
            output = model(data)
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        loss_list.append(run_loss / (i + 1))
        writer.add_scalar('Loss/train', run_loss / (i + 1), epoch)
    print("line done!")
    return loss_list


def one_train(dataloader, device):
    model.to(device)
    run_loss = 0.0
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = data
        data = x.to(device)
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    return run_loss / (i + 1)
