import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

torch.manual_seed(20210831)
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]


class DGI(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(DGI, self).__init__()
        self.encoder = GCNConv(in_features, hidden_features)
        self.decoder = nn.Bilinear(hidden_features, hidden_features, 1)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)  # embeddings
        positive_score = self.decoder(z, z)
        negative_score = self.decoder(z, z[torch.randperm(z.size(0))])  # random shuffle
        return positive_score, negative_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGI(in_features=dataset.num_features, hidden_features=128).to(device)
data = data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    positive_score, negative_score = model(data.x, data.edge_index)
    loss = - (nn.LogSigmoid()(positive_score) + nn.LogSigmoid()(-negative_score)).mean()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
