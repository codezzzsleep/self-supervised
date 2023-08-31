import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SkipGramDataset
from net import SkipGram


def train_skipgram(text, window_size, embedding_dim, batch_size, epochs, learning_rate):
    dataset = SkipGramDataset(text, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = len(set(text))
    model = SkipGram(vocab_size, embedding_dim)
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for center_word, context_word in dataloader:
            model.zero_grad()
            output = model(center_word)

            loss = loss_fn(output, context_word)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss)
        print(f"Epoch: {epoch + 1}, Loss: {total_loss}")

    return model, losses
