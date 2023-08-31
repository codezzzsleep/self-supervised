import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges

# 加载 Cora 数据集
dataset = Planetoid(root='data', name='Cora', transform=train_test_split_edges)
data = dataset[0]


# 定义 GAE 编码器
class GAE_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设定编码器参数，创建 VGAE 实例
out_channels = 16
encoder = GAE_Encoder(dataset.num_features, out_channels)
model = VGAE(encoder).to(device)

# 优化器设置
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练 VGAE
def train():
    model.train()
    optimizer.zero_grad()
    z, mu, logvar = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


# 检查链接预测性能
def eval_auc(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z, _, _ = model.encode(data.x, data.train_pos_edge_index)
    link_probs = model.decoder(z, pos_edge_index, sigmoid=True).squeeze()
    link_neg_probs = model.decoder(z, neg_edge_index, sigmoid=True).squeeze()
    link_probs = torch.cat([link_probs, link_neg_probs], dim=0).cpu()

    link_labels = torch.cat([pos_edge_index.new_ones(pos_edge_index.size(1)),
                             neg_edge_index.new_zeros(neg_edge_index.size(1))], dim=0)

    return roc_auc_score(link_labels, link_probs)


# 训练 GAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
num_epochs = 100
for epoch in range(num_epochs):
    loss = train()
    auc = eval_auc(data.test_pos_edge_index, data.test_neg_edge_index)
    print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')
