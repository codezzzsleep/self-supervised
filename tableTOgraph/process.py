import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load your data (cites, content, paper tables)
# ...
cites_table = pd.read_csv("data/cites.csv")
paper_table = pd.read_csv("data/paper.csv")
content_table = pd.read_csv("data/content.csv")

# Get cited_paper_ids and citing_paper_ids from the cites table
cited_paper_ids = cites_table['cited_paper_id'].values
citing_paper_ids = cites_table['citing_paper_id'].values

# Number of nodes in the graph
num_nodes = len(paper_table)

# Construct the adjacency matrix
adjacency_matrix = np.zeros((num_nodes, num_nodes))
for row, col in zip(cited_paper_ids, citing_paper_ids):
    adjacency_matrix[row, col] = 1
    adjacency_matrix[col, row] = 1

# Normalize the adjacency matrix
adjacency_matrix = sp.coo_matrix(adjacency_matrix)
row_sum = np.array(adjacency_matrix.sum(1))
row_sum_inv = np.power(row_sum, -1).flatten()
row_sum_inv[np.isinf(row_sum_inv)] = 0.
D_inv_sqrt = sp.diags(row_sum_inv)
normalized_adjacency = D_inv_sqrt.dot(adjacency_matrix).dot(D_inv_sqrt)

# Encode node features using one-hot encoding
encoder = OneHotEncoder()
node_features = encoder.fit_transform(content_table['word_cited_id'].values.reshape(-1, 1)).toarray()

# Encode class labels using label encoding
label_encoder = LabelEncoder()
node_labels = label_encoder.fit_transform(paper_table['class_label'])
