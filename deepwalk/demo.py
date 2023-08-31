import torch
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, embedding_size):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_size = embedding_size

    def generate_random_walks(self):
        walks = []
        nodes = list(self.graph.nodes())

        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(node))

        return walks

    def random_walk(self, start_node):
        walk = [start_node]

        while len(walk) < self.walk_length:
            neighbors = list(self.graph.neighbors(walk[-1]))
            if len(neighbors) > 0:
                walk.append(np.random.choice(neighbors))
            else:
                break

        return walk


# 构建图形
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])

# 创建DeepWalk对象
deepwalk = DeepWalk(graph, walk_length=10, num_walks=80, embedding_size=128)

# 生成随机游走序列
walks = deepwalk.generate_random_walks()

# 使用gensim库中的Word2Vec模型学习嵌入
model = Word2Vec(walks, vector_size=deepwalk.embedding_size, window=5, min_count=0, sg=1, workers=4)

# 获取节点嵌入向量
embeddings = model.wv

# 打印节点0的嵌入向量
print(embeddings[0])
