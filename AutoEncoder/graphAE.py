import torch
import torch_geometric.nn as pyg_nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling


# 编码器模型
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# GAE 模型
class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_logits = torch.mm(z, z.t())
        return adj_logits

    def decode(self, z):
        adj_logits = torch.mm(z, z.t())
        return adj_logits

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)


# 数据预处理
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]
x = data.x
edge_index = data.edge_index

# 选择模型参数
in_channels = x.size(1)
hidden_channels = 64
num_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("runs/gae_log")
# 创建模型
model = GAE(in_channels, hidden_channels).to(device)

# 选择损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(num_epochs):
    optimizer.zero_grad()
    adj_logits = model(x, edge_index)

    # 正向边
    pos_edges = edge_index.t()
    pos_weights = torch.ones(pos_edges.size(0), 1)

    # 负向边
    # 负向边
    neg_edges = negative_sampling(edge_index, x.size(0), num_neg_samples=pos_edges.size(0)).t()
    neg_weights = torch.zeros(neg_edges.size(0), 1)

    # 边的组合
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    edge_weights = torch.cat([pos_weights, neg_weights], dim=0)

    # 计算损失
    edge_logits = adj_logits[edges[:, 0], edges[:, 1]].unsqueeze(-1)
    loss = criterion(edge_logits, edge_weights)
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss.item(), epoch)
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# 评估模型
model.eval()
node_embeddings = model.encode(x, edge_index)
reconstructed_adj_matrix = model.decode(node_embeddings)
torch.save(model.state_dict(), "model/last.pth")
