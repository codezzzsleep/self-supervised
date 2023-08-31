import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


# 1. 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, z_dim)
        self.fc2_logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        z_mean = self.fc2_mean(h1)
        z_logvar = self.fc2_logvar(h1)
        return z_mean, z_logvar


# 2. 定义解码器
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        x_reconstructed = torch.sigmoid(self.fc2(h1))
        return x_reconstructed


# 3. 定义 VAE 结构
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_logvar


# 4. 定义重构损失和 KL 散度损失函数
def vae_loss(x_original, x_reconstructed, z_mean, z_logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, x_original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return recon_loss + kl_divergence


# 5. 设置网络和训练参数
input_size = 784
hidden_size = 400
z_dim = 20
output_size = input_size

encoder = Encoder(input_size, hidden_size, z_dim)
decoder = Decoder(z_dim, hidden_size, output_size)
vae = VAE(encoder, decoder)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 50
batch_size = 64

# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 6. 训练网络
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_size)

        optimizer.zero_grad()

        x_reconstructed, z_mean, z_logvar = vae(data)

        loss = vae_loss(data, x_reconstructed, z_mean, z_logvar)

        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs, loss.item()))
