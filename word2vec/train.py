import torch.nn as nn
import torch.optim as optim
import torch


def train(model, data_loader, epochs, learning_rate, device, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    min_loss = 10000.0
    for epoch in range(epochs):
        run_loss = 0.0
        for batch, (center_idx, context_idx) in enumerate(data_loader):
            center_idx = center_idx.to(device)
            context_idx = context_idx.to(device)

            # 修改这一行。将上下文索引移除，输入仅包含中心词索引。
            scores = model(center_idx)
            loss = criterion(scores, context_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            if batch % 500 == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")
                writer.add_scalar("loss/500_batch", loss.item(), batch / 500)
                if loss.item() < min_loss:
                    torch.save(model.state_dict(), 'runs/best_model.pt')

        loss_list.append(run_loss / (batch + 1))
        writer.add_scalar("meansLoss/epoch", run_loss / (batch + 1), epoch)
    return loss_list
