import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import torch.nn.functional as F


# 数据预处理
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def create_vocabulary(tokenized_corpus):
    vocabulary = defaultdict(int)
    for sentence in tokenized_corpus:
        for token in sentence:
            vocabulary[token] += 1
    return vocabulary


def create_context_word_pairs(tokenized_corpus, vocabulary, window_size=2):
    idx_to_word = {i: word for i, word in enumerate(vocabulary)}
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}

    pairs = []
    for sentence in tokenized_corpus:
        indices = [word_to_idx[word] for word in sentence]
        for center_word_pos in range(len(indices)):
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                pairs.append((indices[center_word_pos], indices[context_word_pos]))
    return pairs, word_to_idx, idx_to_word


corpus = [
    "我爱自然语言处理",
    "自然语言处理是非常有趣的",
    # add more sentences
]

tokenized_corpus = tokenize_corpus(corpus)
vocabulary = create_vocabulary(tokenized_corpus)
pairs, word_to_idx, idx_to_word = create_context_word_pairs(tokenized_corpus, vocabulary, window_size=2)


def negative_sampling_distribution(vocabulary):
    word_freqs = np.array([freq for _, freq in vocabulary.items()], dtype=np.float64)
    unigram_distribution = word_freqs / np.sum(word_freqs)
    return torch.tensor(unigram_distribution ** (3 / 4))


class NegativeSamplingWord2VecDataset(Dataset):
    def __init__(self, pairs, neg_dist, num_neg_samples=5):
        self.pairs = pairs
        self.neg_dist = neg_dist
        self.num_neg_samples = max(1, num_neg_samples)  # 修复: 负采样数限制最小为1。

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        neg_samples = torch.multinomial(self.neg_dist, self.num_neg_samples, replacement=True)
        return center, context, neg_samples


neg_dist = negative_sampling_distribution(vocabulary)
dataset = NegativeSamplingWord2VecDataset(pairs, neg_dist)


# 构建模型
class NegativeSamplingWord2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(NegativeSamplingWord2Vec, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, center, context, neg_samples):
        center_embed = self.in_embed(center)
        context_embed = self.out_embed(context)
        neg_embed = self.out_embed(neg_samples)

        positive_score = torch.bmm(center_embed.unsqueeze(1), context_embed.unsqueeze(2)).squeeze()
        negative_score = torch.bmm(center_embed.unsqueeze(1), neg_embed.permute(0, 2, 1)).squeeze()

        loss = -torch.sum(F.logsigmoid(positive_score) + torch.sum(F.logsigmoid(-negative_score), dim=1))
        return loss


vocab_size = len(vocabulary)
embed_size = 100
model = NegativeSamplingWord2Vec(vocab_size, embed_size)

# 训练模型
epochs = 100
batch_size = 64
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model = model.to(device)

for epoch in range(epochs):
    losses = []
    for batch in dataloader:
        center, context, neg_samples = batch
        center, context, neg_samples = center.to(device), context.to(device), neg_samples.to(device)

        optimizer.zero_grad()
        loss = model(center, context, neg_samples)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Epoch {epoch + 1}, Loss: {np.mean(losses)}")
