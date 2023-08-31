import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

# 定义期望平均激活值和KL散度的权重
expect_tho = 0.05
hidden_size = 30
tho_tensor = torch.FloatTensor([expect_tho for _ in range(hidden_size)])
if torch.cuda.is_available():
    tho_tensor = tho_tensor.cuda()
_beta = 3


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


def KL_devergence(p, q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0) / batch_size  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=784, hidden_size=30, out_dim=784):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


# 3. 定义稀疏损失函数
def sparse_loss(hidden_activation, target_sparsity, sparsity_weight):
    epsilon = 1e-10
    mean_activation = torch.mean(hidden_activation, dim=0)
    kl_divergence = target_sparsity * torch.log(target_sparsity / (mean_activation + epsilon)) + \
                    (1 - target_sparsity) * torch.log((1 - target_sparsity) / (1 - mean_activation + epsilon))
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
