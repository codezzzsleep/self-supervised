import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 超参数
batch_size = 256
learning_rate = 0.001
num_epochs = 200

# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 堆叠自编码器模型
class StackedAutoencoder(nn.Module):
    def __init__(self):
        super(StackedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StackedAutoencoder().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter("runs/stack_log")

step = 2
# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, images)
        running_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        if epoch == step:
            writer.add_image(f"imgae_{step}", images[23].view(1, 28, 28), step)
            writer.add_image(f"output_image_{step}", outputs[23].view(1, 28, 28), step)
            step = step * 2
    writer.add_scalar("Loss_epoch/train", running_loss / i, epoch + 1)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
writer.close()
print('Training completed.')
