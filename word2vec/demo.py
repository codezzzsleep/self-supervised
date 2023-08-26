import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


# 定义模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, target, context):
        embed_target = self.in_embed(target)
        embed_context = self.out_embed(context)

        score = torch.mul(embed_target, embed_context)
        score = torch.sum(score, dim=1)
        log_probs = F.log_softmax(score).view(-1)

        return log_probs


# 假设我们有一些预处理的数据
vocab_size = 1000
embed_size = 100
learning_rate = 0.001
epochs = 10

# 初始化模型和优化器
model = SkipGramModel(vocab_size, embed_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(epochs):
    total_loss = 0
    for batch in range(100):  # 假设我们有100个batch
        target, context = get_batch()  # 你需要定义get_batch()来获取数据
        target = Variable(torch.LongTensor(target))
        context = Variable(torch.LongTensor(context))
        model.zero_grad()
        log_probs = model(target, context)
        loss = F.nll_loss(log_probs, Variable(torch.LongTensor(context)))
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    print("Epoch: " + str(epoch) + ", Loss: " + str(total_loss / 100))
