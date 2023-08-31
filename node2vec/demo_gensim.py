import random

import networkx as nx
from gensim.models import Word2Vec


class Node2Vec:

    def __init__(self, graph, p, q, walk_length, num_walks, embed_size, iterations, window_size):
        self.graph = graph
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embed_size = embed_size
        self.iterations = iterations
        self.window_size = window_size
        self.walks = []

    def node2vec_walk(self, start_node):
        walk = [start_node]

        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(walk[-1]))
            if len(neighbors) == 0:
                break

            candidates = []
            for neighbor in neighbors:
                if len(walk) == 1:
                    candidates.append(neighbor)
                else:
                    last_node = walk[-2]
                    weight = self.transition_weight(last_node, walk[-1], neighbor)
                    candidates.extend([neighbor] * weight)

            chosen_node = random.choice(candidates)
            walk.append(chosen_node)
        return walk

    def transition_weight(self, prev_node, current_node, next_node):
        if next_node == prev_node:
            return 1 / self.p
        elif self.graph.has_edge(next_node, prev_node):
            return 1
        else:
            return 1 / self.q

    def create_random_walks(self):
        for _ in range(self.num_walks):
            nodes = list(self.graph.nodes())
            random.shuffle(nodes)

            for node in nodes:
                walk = self.node2vec_walk(node)
                self.walks.append(walk)

    def learn_embedding(self):
        str_walks = [[str(node) for node in walk] for walk in self.walks]
        model = Word2Vec(str_walks, vector_size=self.embed_size, window=self.window_size, epochs=self.iterations)
        node_embeddings = {str(node): model.wv[str(node)] for node in self.graph.nodes()}
        return node_embeddings


def main():
    G = nx.karate_club_graph()  # 创建或加载图
    p, q = 1, 0.5
    walk_length = 10
    num_walks = 10
    embed_size = 128
    iterations = 5
    window_size = 5

    node2vec = Node2Vec(G, p, q, walk_length, num_walks, embed_size, iterations, window_size)
    node2vec.create_random_walks()
    embeddings = node2vec.learn_embedding()

    print("Embeddings:")
    for node, embedding in embeddings.items():
        print("Node: {}, Embedding: {}".format(node, embedding))


if __name__ == '__main__':
    main()
