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


# 建立 ConvolutionalAutoencoder 模型
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvolutionalAutoencoder().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter("runs/conv_log")

step = 2
# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, images)
        running_loss += loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == step:
            writer.add_image(f"image_{step}", images[23], step)
            writer.add_image(f"output_image_{step}", outputs[23], step)
            step = step * 2

    writer.add_scalar("Loss_epoch/train", running_loss / i, epoch + 1)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

writer.close()
print('Training completed.')
