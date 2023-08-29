import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

cites_table = pd.read_csv("data/cites.csv")
paper_table = pd.read_csv("data/paper.csv")
content_table = pd.read_csv("data/content.csv")

# 获取cited_paper_ids 和 citing_paper_ids
cited_paper_ids = cites_table['cited_paper_id'].values
citing_paper_ids = cites_table['citing_paper_id'].values

max_paper_id = max(cites_table['cited_paper_id'].max(),
                   cites_table['citing_paper_id'].max()) + 1
min_paper_id = min(cites_table['cited_paper_id'].min(),
                   cites_table['citing_paper_id'].min())
# print(max_paper_id)
# 1155073
# adjacency_matrix = np.zeros((max_paper_id, max_paper_id))
# numpy.core._exceptions.MemoryError:
# Unable to allocate 9.71 TiB for an array with shape (1155073, 1155073)
# and data type float64
# print(min_paper_id)
# 35

# 使用 1 进行作为连接的值
data = np.ones_like(cited_paper_ids)
adjacency_matrix = sp.coo_matrix((data, (cited_paper_ids, citing_paper_ids)),
                                 shape=(max_paper_id, max_paper_id))

print(len(adjacency_matrix.data))
# adjacency_matrix = np.zeros((max_paper_id, max_paper_id))
# for row, col in zip(cited_paper_ids, citing_paper_ids):
#     adjacency_matrix[row, col] = 1
#     adjacency_matrix[col, row] = 1
# adjacency_matrix = sp.coo_matrix(adjacency_matrix)

# Normalize the adjacency matrix
# row_sum = np.array(adjacency_matrix.sum(1))
# row_sum = row_sum.astype(float)  # 转换为浮点数
# row_sum_inv = np.zeros_like(row_sum)  # 创建一个填充为零的相同形状的数组
#
# # 查找 row_sum 中非零的元素
# nonzero_mask = row_sum != 0
#
# # 仅计算非零元素的倒数
# row_sum_inv[nonzero_mask] = np.power(row_sum[nonzero_mask], -1).flatten()
#
# D_inv_sqrt = sp.diags(row_sum_inv)
# normalized_adjacency = D_inv_sqrt.dot(adjacency_matrix).dot(D_inv_sqrt)

# Encode node features using one-hot encoding
encoder = OneHotEncoder()
node_features = encoder.fit_transform(content_table['word_cited_id'].values.reshape(-1, 1)).toarray()

# Encode class labels using label encoding
label_encoder = LabelEncoder()
node_labels = label_encoder.fit_transform(paper_table['class_label'])
print(node_features)
print(node_features.shape)
print(1)
print(node_labels)
print(len(node_labels))
print(max(node_labels))
