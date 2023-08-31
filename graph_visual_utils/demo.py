import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from(range(5))
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

pos_random = nx.random_layout(G, seed=42)
pos_circular = nx.circular_layout(G)
pos_kamada_kawai = nx.kamada_kawai_layout(G)
pos_spectral = nx.spectral_layout(G)

layouts = [
    ("Random", pos_random),
    ("Circular", pos_circular),
    ("Kamada-Kawai", pos_kamada_kawai),
    ("Spectral", pos_spectral),
]

num_layouts = len(layouts)

# 画图，显示不同布局
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, (title, layout) in enumerate(layouts):
    ax = axes[i]

    nx.draw(
        G,
        layout,
        with_labels=True,
        node_color="skyblue",
        node_size=1500,
        width=3,
        edge_color="grey",
        alpha=0.8,
        ax=ax,
    )

    ax.set_title(title)
    ax.axis("off")

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
