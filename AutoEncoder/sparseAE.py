import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


# 1. 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


# 2. 定义解码器
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


# 3. 定义稀疏损失函数
def sparse_loss(hidden_activation, target_sparsity, sparsity_weight):
    mean_activation = torch.mean(hidden_activation, dim=0)
    kl_divergence = target_sparsity * torch.log(target_sparsity / mean_activation) + \
                    (1 - target_sparsity) * torch.log((1 - target_sparsity) / (1 - mean_activation))
    return sparsity_weight * torch.sum(kl_divergence)


# 4. 设置网络和训练参数
input_size = 784
hidden_size = 100
output_size = input_size

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.MSELoss()

epochs = 100
batch_size = 64
target_sparsity = 0.05
sparsity_weight = 1e-5
# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 5. 训练网络
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_size)

        optimizer.zero_grad()

        hidden_representation = encoder(data)
        reconstructed_data = decoder(hidden_representation)

        mse_loss = criterion(reconstructed_data, data)
        sparsity_penalty = sparse_loss(hidden_representation, target_sparsity, sparsity_weight)
        loss = mse_loss + sparsity_penalty

        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs, loss.item()))
