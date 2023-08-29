import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections.abc import Iterable

edge = pd.read_csv("data/cites.csv")
lable = pd.read_csv("data/paper.csv")
vertice = pd.read_csv("data/content.csv")


def create_graph_from_table(vertices_data, edge, lable, normalize=False):
    head, tail = edge
    vertice, data = vertices_data

    if isinstance(head, Iterable) is True:
        m = max(max(head), max(tail)) + 1
    else:
        m = max(head, tail) + 1
    weight = np.ones_like(head)
    adjacency_matrix = sp.coo_matrix((weight, (head, tail)),
                                     shape=(m, m))
    if normalize is True:
        pass
    # Encode node features using one-hot encoding
    encoder = OneHotEncoder()
    node_features = encoder.fit_transform(content_table['word_cited_id'].values.reshape(-1, 1)).toarray()

    # Encode class labels using label encoding
    label_encoder = LabelEncoder()
    node_labels = label_encoder.fit_transform(paper_table['class_label'])


if __name__ == "__main__":
    create_graph_from_table(1, ([1, 2], [2, 6]), 1)
