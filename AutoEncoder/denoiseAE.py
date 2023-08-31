import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


# 定义去噪自动编码器
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 定义添加噪声的函数
def add_noise(images, noise_factor=0.5):
    images = images + noise_factor * torch.randn(*images.shape)
    images = images.clip(0., 1.)
    return images


# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = MNIST('data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 构建和训练去噪自动编码器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for images, _ in train_loader:
        noisy_images = add_noise(images)
        noisy_images, images = noisy_images.to(device), images.to(device)

        outputs = autoencoder(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
