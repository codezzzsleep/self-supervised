import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embed_size)
        self.out_embedding = nn.Linear(embed_size, vocab_size)

    def forward(self, center_idx):
        center_embed = self.in_embedding(center_idx)
        scores = self.out_embedding(center_embed)
        return scores


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(CBOW, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_word_indexes):
        embeddings = torch.mean(self.embedding_layer(context_word_indexes), dim=1)
        h1 = torch.relu(self.linear1(embeddings))
        out = self.linear2(h1)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word):
        embeds = self.embeddings(center_word)
        out = self.linear(embeds)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs
