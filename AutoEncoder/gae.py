import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x


class GraphDecoder(nn.Module):
    def __init__(self):
        super(GraphDecoder, self).__init__()

    def forward(self, z):
        z_norm = torch.norm(z, p=2, dim=-1).unsqueeze(-1)
        z_sim = z @ z.t() / (z_norm * z_norm.t())
        return torch.sigmoid(z_sim)


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, embedding_dim)
        self.decoder = GraphDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred


dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# 为了训练图自编码器，我们需要通过移除部分边来创建训练和验证集
data = train_test_split_edges(data)

# 设定模型参数
input_dim = dataset.num_features
hidden_dim = 128
embedding_dim = 64

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphAutoencoder(input_dim, hidden_dim, embedding_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 训练函数
def train():
    model.train()
    optimizer.zero_grad()

    z = model.encoder(data.x.to(device), data.train_pos_edge_index.to(device))
    adj_pred = model.decoder(z)
    adj_true = data.train_pos_edge_adj_t.to_dense().to(device)

    loss = criterion(adj_pred, adj_true)
    loss.backward()
    optimizer.step()

    return loss.item()


# 训练模型
epochs = 100
for epoch in range(1, epochs + 1):
    loss = train()
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")
