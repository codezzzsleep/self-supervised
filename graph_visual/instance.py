from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import networkx as nx
import matplotlib.pyplot as plt
import torch

# 从 PyG 加载 Cora 数据集
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# 导出 PyG 数据为 NetworkX 有向图
data = dataset[0]
edges = data.edge_index.t().tolist()
graph = nx.DiGraph()
graph.add_edges_from(edges)

# If you would like to convert it to an undirected graph, you can use this
# graph = graph.to_undirected()

# 可视化
nx.draw(
    graph,
    node_color='skyblue',
    edge_color='grey',
    node_size=30,
    width=0.2,
)
plt.show()
