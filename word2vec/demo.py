import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 定义Word2Vec模型类
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embeds = self.embeddings(target)
        scores = self.linear1(embeds)
        log_probs = nn.functional.log_softmax(scores, dim=1)
        return log_probs

# 训练函数
def train(model, data, optimizer):
    model.train()
    total_loss = 0

    for context, target in data:
        context_var = Variable(torch.LongTensor(context))
        target_var = Variable(torch.LongTensor(target))  # 修改此处

        optimizer.zero_grad()
        log_probs = model(context_var)

        loss = nn.functional.nll_loss(log_probs, target_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    return total_loss.item() / len(data)

# 准备数据
data = [
    ([1, 2], [3]),  # 修改此处
    ([2, 3], [4]),  # 修改此处
    ([3, 4], [5]),  # 修改此处
    ([4, 5], [6]),  # 修改此处
    ([5, 6], [7])   # 修改此处
]

vocab_size = 8
embedding_dim = 3
model = Word2Vec(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, data, optimizer)
    print(f"Epoch: {epoch+1}, Loss: {loss}")
