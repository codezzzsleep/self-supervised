import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

writer = SummaryWriter("runs/gvs_log")

dataset = Planetoid(root='data', name='Cora')
data = dataset[0]


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


# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.softmax(x, dim=1)


class GAEGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GAEGCN, self).__init__()
        self.conv = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.softmax(x, dim=1)


# 创建 GCN 实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
gaeGCN = GAEGCN(64, dataset.num_classes).to(device)
data = dataset[0].to(device)
gcn_optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
gcn_gae_optimizer = torch.optim.Adam(gaeGCN.parameters(), lr=0.01)
gae = GAE(1433, 64).to(device).to(device)
gae.load_state_dict(torch.load("model/last.pth"))

writer = SummaryWriter("runs/vs_log")
# 训练模型
gae.eval()
gcn.train()
gaeGCN.train()
for epoch in range(200):
    gcn_optimizer.zero_grad()
    gcn_gae_optimizer.zero_grad()
    gcn_out = gcn(data)
    t_data = gae.encode(data.x, data.edge_index)
    gae_gcn_out = gaeGCN(t_data, data.edge_index)
    gcn_loss = F.nll_loss(gcn_out[data.train_mask], data.y[data.train_mask])
    gae_gcn_loss = F.nll_loss(gae_gcn_out[data.train_mask], data.y[data.train_mask])
    gcn_loss.backward()
    gae_gcn_loss.backward()
    gcn_optimizer.step()
    gcn_gae_optimizer.step()
    writer.add_scalar("Loss/train", gcn_loss.item(), epoch)
    writer.add_scalar("Loss/train_gae", gae_gcn_loss.item(), epoch)
    # 输出损失
    if (epoch + 1) % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(
            epoch + 1, 200, gcn_loss.item()))

# 评估模型
gcn.eval()
gaeGCN.eval()
_, gcn_pred = gcn(data).max(dim=1)
t_data = gae.encode(data.x, data.edge_index)
_, gcn_gae_pred = gaeGCN(t_data, data.edge_index).max(dim=1)
gcn_correct = int(gcn_pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
gcn_gae_correct = int(gcn_gae_pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())

gcn_accuracy = gcn_correct / int(data.test_mask.sum())
gcn_gae_accuracy = gcn_gae_correct / int(data.test_mask.sum())

print('GCN Accuracy: {:.4f}'.format(gcn_accuracy))
print('gaeGCn Accuracy: {:.4f}'.format(gcn_gae_accuracy))

writer.close()
