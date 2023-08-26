import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter


# 构建词汇表和数据集
class CBOWDataset(Dataset):
    def __init__(self, text, window_size=2):
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


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        embedding = torch.mean(self.embeddings(inputs), dim=1)
        logits = self.linear(embedding)
        return logits


def train(model, data_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        run_loss = 0.0
        for batch, (center_idx, context_idx) in enumerate(data_loader):
            context_idx = context_idx.to(device)
            center_idx = center_idx.to(device)

            logits = model(context_idx.view(-1, 1))

            loss = criterion(logits, center_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        run_loss /= (batch + 1)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {run_loss}")


file_path = 'data/text8.train.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

window_size = 2
embed_size = 50
batch_size = 4
epochs = 1
learning_rate = 0.01

dataset = CBOWDataset(text, window_size=window_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(dataset.word_to_idx)
model = CBOW(vocab_size, embed_size)
model.to(device)

train(model, data_loader, epochs, learning_rate)
