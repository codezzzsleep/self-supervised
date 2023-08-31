import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 加载 Cora 数据集
dataset = Planetoid(root='data', name='Cora')


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


# 创建 GCN 实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
writer = SummaryWriter("runs/gcn_log")
# 训练模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss.item(), epoch)
    # 输出损失
    if (epoch + 1) % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(
            epoch + 1, 200, loss.item()))

# 评估模型
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(accuracy))
torch.save(model.state_dict(), "model/gcn.pth")
writer.close()
