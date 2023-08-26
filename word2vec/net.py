import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embed_size)
        self.out_embedding = nn.Linear(embed_size, vocab_size)

    def forward(self, center_idx):
        center_embed = self.in_embedding(center_idx)
        scores = self.out_embedding(center_embed)
        return scores
