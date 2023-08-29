import networkx as nx
import matplotlib.pyplot as plt

cited_paper_ids = [1, 2, 1, 3, 1, 4, 3, 5]
citing_paper_ids = [2, 1, 3, 1, 4, 1, 5, 3]

G = nx.DiGraph()

# 添加节点，确保所有提到的论文ID都被添加
nodes = set(cited_paper_ids + citing_paper_ids)
G.add_nodes_from(nodes)

# 添加有向边
for i, j in zip(cited_paper_ids, citing_paper_ids):
    G.add_edge(j, i)  # j 为引用的论文，i 为被引用的论文

# 绘制有向图
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='grey', width=2, alpha=0.8)
plt.show()
