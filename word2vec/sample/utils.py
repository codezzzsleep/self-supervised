from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from dataset import SkipGramDataset
import matplotlib.pyplot as plt


def load_data(text=None, window_size=None, batch_size=None):
    if text is None:
        text = "this is a simple example for word2vec using pytorch"
    if window_size is None:
        window_size = 2
    if batch_size is None:
        batch_size = 256
    dataset = SkipGramDataset(text, window_size=window_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, data_loader


def plot_with_labels(low_dim_embs, labels, filename='TSNE_result.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


# 使用T-SNE算法将128维降低到2维
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=1)
# 绘制点的个数
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[: plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
plt.show()
