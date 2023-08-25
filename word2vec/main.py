import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np


# 构建词汇表和数据集
class SkipGramDataset(Dataset):
    def __init__(self, text, window_size=2):
        # 对文本进行处理，生成中心词和背景词对
        tokenized_text = text.split()
        word_counts = Counter(tokenized_text)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.window_size = window_size
        self.data = []

        for idx, center_word in enumerate(tokenized_text):
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                context_idx = idx + offset
                if context_idx < 0 or context_idx >= len(tokenized_text):
                    continue

                context_word = tokenized_text[context_idx]
                self.data.append((center_word, context_word))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_word, context_word = self.data[idx]
        center_idx = self.word_to_idx[center_word]
        context_idx = self.word_to_idx[context_word]
        return center_idx, context_idx


# 定义 Word2Vec 模型
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embed_size)
        self.out_embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, center_idx, context_idx):
        center_embed = self.in_embedding(center_idx)
        context_embed = self.out_embedding(context_idx)
        scores = torch.mul(center_embed, context_embed).sum(dim=-1)
        return scores


# 训练函数
def train(model, data_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch, (center_idx, context_idx) in enumerate(data_loader):
            center_idx = center_idx.to(device)
            context_idx = context_idx.to(device)

            scores = model(center_idx, context_idx)
            loss = criterion(scores, context_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")


# 示例文本
text = "this is a simple example for word2vec using pytorch"

# 超参数设置
window_size = 2
embed_size = 50
batch_size = 4
epochs = 10
learning_rate = 0.01

# 数据准备
dataset = SkipGramDataset(text, window_size=window_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 Word2Vec 模型并将其移至设备（当有 GPU 可用时）
vocab_size = len(dataset.word_to_idx)
model = Word2Vec(vocab_size, embed_size)
model.to(device)

# 训练模型
train(model, data_loader, epochs, learning_rate)
