import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 示例文本
text = """we are about to study the idea of a computational process. computational processes are abstract
    beings that inhabit computers. as they evolve, processes manipulate other abstract
    things called data. the evolution of a process is directed by a pattern of rules
    called a program."""

# 分词，得到单词集合
words = text.lower().split()

# 创建词汇表及索引与单词的映射
vocabulary = set(words)
word2index = {word: i for i, word in enumerate(vocabulary)}
index2word = {i: word for i, word in enumerate(vocabulary)}

# 创建训练数据集 (中心词, 上下文词)
window_size = 2
data = []
for i, word in enumerate(words):
    for j in range(max(i - window_size, 0), min(i + window_size + 1, len(words))):
        if j != i:
            data.append((word, words[j]))


class DatasetModel(Dataset):
    def __init__(self, data, word2index):
        self.data = data
        self.word2index = word2index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_word, context_word = self.data[idx]
        return torch.tensor(self.word2index[center_word]), torch.tensor(self.word2index[context_word])


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context):
        in_embeds = self.in_embed(target)
        out_embeds = self.out_embed(context)

        scores = torch.matmul(in_embeds, out_embeds.t())

        return scores


def train(dataset, model, optimizer, loss_fn, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (center_word, context_word) in enumerate(dataloader):
            optimizer.zero_grad()
            scores = model(center_word, context_word)

            loss = loss_fn(scores, context_word)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")


vocab_size = len(vocabulary)
embed_dim = 20
learning_rate = 0.001
epochs = 100
batch_size = 32

dataset = DatasetModel(data, word2index)
model = SkipGramModel(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

train(dataset, model, optimizer, loss_fn, epochs, batch_size)
