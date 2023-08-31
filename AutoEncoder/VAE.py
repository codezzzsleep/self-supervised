import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义变分自动编码器的网络架构
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        mean = self.fc2_mean(hidden)
        logvar = self.fc2_logvar(hidden)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        output = torch.sigmoid(self.fc2(hidden))
        return output


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar


# 定义损失函数
def vae_loss(reconstruction, x, mean, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstruction, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


# 超参数
input_dim = 784  # 输入数据的维度
hidden_dim = 256  # 隐藏层维度
latent_dim = 20  # 潜在变量维度
batch_size = 128
num_epochs = 10

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建 VAE 模型和优化器
vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 训练 VAE 模型
for epoch in range(num_epochs):
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(-1, input_dim)
        optimizer.zero_grad()
        reconstruction, mean, logvar = vae(x)
        loss = vae_loss(reconstruction, x, mean, logvar)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")


