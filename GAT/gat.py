import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.transforms import NormalizeFeatures


# 定义 GAT 层
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.6):
        super(GATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


# 定义 GAT 模型
class GATModel(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_heads, alpha=0.2, dropout=0.6):
        super(GATModel, self).__init__()

        self.heads = n_heads
        self.GAT_layers = nn.ModuleList(
            [GATLayer(in_features=n_feat, out_features=n_hid, alpha=alpha, dropout=dropout) for _ in range(n_heads)])
        self.final_layer = GATLayer(in_features=n_heads * n_hid, out_features=n_classes, alpha=alpha, dropout=dropout)

    def forward(self, x, adj):
        x = torch.cat([gat_layer(x, adj) for gat_layer in self.GAT_layers], dim=1)
        x = F.elu(x)
        x = F.dropout(x, p=self.GAT_layers[0].dropout, training=self.training)
        x = self.final_layer(x, adj)

        return F.log_softmax(x, dim=1)


# 数据加载与预处理
dataset = Planetoid(root="~/cora", name="Cora", transform=NormalizeFeatures())
data = dataset[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
adj = to_dense_adj(add_self_loops(data.edge_index)[0])[0].to(device)

# 实例化 GAT 模型
n_feat = dataset.num_node_features
n_hid = 8
n_classes = dataset.num_classes
n_heads = 8

model = GATModel(n_feat, n_hid, n_classes, n_heads).to(device)
loss_func = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# 训练循环
def train_loop(epoch, model, optimizer, data, adj, loss_func):
    model.train()
    optimizer.zero_grad()

    output = model(data.x, adj)
    loss = loss_func(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_acc = accuracy(output, data.train_mask)
    val_acc = accuracy(output, data.val_mask)

    print("Epoch: {:03d}".format(epoch), end=" - ")
    print("Train Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}".format(loss.item(), train_acc, val_acc))


# 计算准确率
def accuracy(output, mask):
    pred = output[mask].max(dim=1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc


# 训练 GAT 模型
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    train_loop(epoch, model, optimizer, data, adj, loss_func)

# 测试
model.eval()
output = model(data.x, adj)
test_acc = accuracy(output, data.test_mask)
print("Test Accuracy: {:.4f}".format(test_acc))
