import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class SkipGramDataset(Dataset):
    def __init__(self, text, window_size=2):
        tokenized_text = text.split()
        word_counts = Counter(tokenized_text)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        word_freqs = np.asarray(sorted(word_counts.values(), reverse=True))
        self.word_probs = word_freqs / word_freqs.sum()

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


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size, negative_samples=5):
        super(Word2Vec, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embed_size)
        self.out_embedding = nn.Linear(embed_size, vocab_size)
        self.negative_samples = negative_samples

    def negative_sampling(self, scores, context_idx, word_probs):
        n = self.negative_samples
        n_samples = torch.tensor(
            np.random.choice(len(self.word_to_idx), (scores.shape[0], n), replace=True, p=word_probs)).to(device)

        out_embeddings = self.out_embedding.weight

        true_scores = torch.gather(scores, 1, context_idx.unsqueeze(1)).squeeze()
        true_scores_term = torch.log(torch.sigmoid(true_scores))

        negative_scores = scores.gather(1, n_samples)
        negative_scores_term = torch.sum(torch.log(torch.sigmoid(-1 * negative_scores)), dim=1)

        return -torch.mean(true_scores_term + negative_scores_term)

    def forward(self, center_idx, context_idx, word_probs):
        center_embed = self.in_embedding(center_idx)
        scores = self.out_embedding(center_embed)
        if self.negative_samples > 0:
            loss = self.negative_sampling(scores, context_idx, word_probs)
        else:
            loss = criterion(scores, context_idx)
        return loss


loss_list = []
writer = SummaryWriter("runs/logs")
min_loss = 10000.0
criterion = nn.CrossEntropyLoss()


def train(model, data_loader, dataset, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        run_loss = 0.0
        for batch, (center_idx, context_idx) in enumerate(data_loader):
            center_idx = center_idx.to(device)
            context_idx = context_idx.to(device)

            # 修改这一行，将上下文索引和词频分布作为输入
            loss = model(center_idx, context_idx, dataset.word_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            if batch % 500 == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")
                writer.add_scalar("loss/500_batch", loss.item(), batch / 500)

                if loss.item() < min_loss:
                    torch.save(model.state_dict(), 'runs/best_model.pt')

        loss_list.append(run_loss / (batch + 1))
        writer.add_scalar("meansLoss/epoch", run_loss / (batch + 1), epoch)


file_path = 'data/text8.train.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

window_size = 2
embed_size = 50
negative_samples = 5
batch_size = 4
epochs = 1
learning_rate = 0.01

dataset = SkipGramDataset(text, window_size=window_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(dataset.word_to_idx)
model = Word2Vec(vocab_size, embed_size, negative_samples)
model.to(device)

train(model, data_loader, dataset, epochs, learning_rate)
print("train done!")
torch.save(model.state_dict(), 'runs/last_model.pt')
writer.close()
x = list(range(1, epochs + 1))
plt.plot(x, loss_list)

plt.title('loss/epoch')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.show()
print("task done!")
