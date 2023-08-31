import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

# 数据准备
dataset = Planetoid(root='data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]


# 构建GAE网络
class GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            GCNConv(in_channels, 2 * out_channels, cached=True),
            torch.nn.ReLU(),
            GCNConv(2 * out_channels, out_channels, cached=True)
        )

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(dataset.num_features, 32).to(device)

# 将数据拆分为训练和测试数据集
data = train_test_split_edges(data)

# 定义损失和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练过程
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x.to(device), data.train_pos_edge_index.to(device))
    loss = -torch.mean(z[data.train_pos[0]] * z[data.train_pos[1]]).sum()
    loss.backward()
    optimizer.step()
    return loss.item()


# 评估过程
def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.train_pos_edge_index.to(device))

    pos_pred = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1))
    neg_pred = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1))
    y_true = torch.cat([pos_pred.new_ones(pos_pred.shape[0]), neg_pred.new_zeros(neg_pred.shape[0])], dim=-1)
    y_pred = torch.cat([pos_pred, neg_pred], dim=-1)
    return torch.eq(y_true, y_pred > 0.5).sum().item() / y_true.size(0)


# 模型训练与评估
for epoch in range(1, 201):
    loss = train()
    auc_train = test(data.train_pos_edge_index, data.train_neg_edge_index)
    auc_test = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, Loss: {:.4f}, AUC Train: {:.4f}, AUC Test: {:.4f}'.format(epoch, loss, auc_train, auc_test))
