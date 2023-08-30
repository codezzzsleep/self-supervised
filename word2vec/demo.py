import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
from collections import Counter
import random

# 超参数设置
context_size = 2
embedding_dim = 100
epochs = 10
batch_size = 64

# 你的语料库
corpus = ['This', 'is', 'an', 'example', 'sentence']  # 替换为实际文本数据，如['This', 'is', 'an', 'example', 'sentence']


# 构建词汇表
def build_vocab(corpus, min_freq):
    word_count = Counter(corpus)
    vocab = [word for word, freq in word_count.items() if freq >= min_freq]
    return vocab


# 生成上下文词对
def generate_context_pairs(corpus, context_size):
    context_pairs = []
    for idx in range(context_size, len(corpus) - context_size):
        center_word = corpus[idx]
        context_words = corpus[idx - context_size:idx] + corpus[idx + 1:idx + context_size + 1]
        context_pairs.extend([(center_word, context) for context in context_words])
    return context_pairs


# Skip-gram模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_word, context_word):
        center_embeds = self.center_embeddings(center_word)
        context_embeds = self.context_embeddings(context_word)

        scores = torch.matmul(center_embeds, context_embeds.t())
        return scores


# CBOW模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_words):
        embeds = torch.mean(self.embeddings(context_words), dim=1)
        scores = torch.matmul(embeds, self.embeddings.weight.t())
        return scores


def train(model, dataloader, optimizer, loss_fn, device, vocab_size):
    model.train()
    for batch, (center_word, context_word) in enumerate(dataloader):
        center_word, context_word = center_word.to(device), context_word.to(device)
        optimizer.zero_grad()

        if isinstance(model, SkipGram):
            scores = model(center_word, context_word)
            context_word = torch.tensor([context_word[i // vocab_size] for i in range(scores.size(0))],
                                        dtype=torch.long)
        elif isinstance(model, CBOW):
            context_word_list = []
            for idx, center in enumerate(center_word):
                context_word_list.append(
                    torch.tensor([context_word[item] for item in range(idx, context_word.size(0), len(center_word))],
                                 dtype=torch.long).to(device))
            context_words_batch = torch.stack(context_word_list)
            scores = model(context_words_batch)

        loss = loss_fn(scores, center_word if isinstance(model, CBOW) else context_word)
        loss.backward()
        optimizer.step()

    # 数据准备


vocab = build_vocab(corpus, min_freq=1)
word2index = {word: i for i, word in enumerate(vocab)}
index2word = {i: word for word, i in word2index.items()}
vocab_size = len(vocab)

context_pairs = generate_context_pairs(corpus, context_size)
center_word_data = [word2index[pair[0]] for pair in context_pairs]
context_word_data = [word2index[pair[1]] for pair in context_pairs]

# 数据集准备
dataset = data.TensorDataset(torch.tensor(center_word_data, dtype=torch.long),
                             torch.tensor(context_word_data, dtype=torch.long))
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型、优化器、损失函数准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skipgram_model = SkipGram(vocab_size, embedding_dim).to(device)
cbow_model = CBOW(vocab_size, embedding_dim).to(device)
optimizer_skipgram = optim.Adam(skipgram_model.parameters(), lr=0.001)
optimizer_cbow = optim.Adam(cbow_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    print("Training Skip-gram model...")
    train(skipgram_model, dataloader, optimizer_skipgram, loss_fn, device, vocab_size)

    print("Training CBOW model...")
    train(cbow_model, dataloader, optimizer_cbow, loss_fn, device, vocab_size)
