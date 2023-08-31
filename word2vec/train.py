import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from word2vec.sample.net import CBOW, SkipGram

torch.manual_seed(1)

CONTEXT_SIZE = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'data/text8'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read().split()
vocab = set(text)
vocab_size = len(vocab)
print('vocab_size:', vocab_size)

# 使用字典推导式
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}


# context window size is two
def create_cbow_dataset(text):
    data = []
    for i in range(2, len(text) - 2):
        context = [text[i - 2], text[i - 1],
                   text[i + 1], text[i + 2]]
        target = text[i]
        data.append((context, target))
    return data


def create_skipgram_dataset(text):
    import random
    data = []
    for i in range(2, len(text) - 2):
        data.append((text[i], text[i - 2], 1))
        data.append((text[i], text[i - 1], 1))
        data.append((text[i], text[i + 1], 1))
        data.append((text[i], text[i + 2], 1))
        # 随机负采样 这里是4个，但是 应该为一个 超参数
        for _ in range(4):
            if random.random() < 0.5 or i >= len(text) - 3:
                rand_id = random.randint(0, i - 1)
            else:
                rand_id = random.randint(i + 3, len(text) - 1)
            data.append((text[i], text[rand_id], 0))
    return data


class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(2 * context_size * embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs).view((1, -1))
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)

    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1))
        embed_ctx = self.embeddings(context).view((1, -1))
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = F.logsigmoid(score)

        return log_probs


cbow_train_data = create_cbow_dataset(text)
skipgram_train_data = create_skipgram_dataset(text)

embd_size = 50
learning_rate = 0.001
n_epoch = 30
writer = SummaryWriter("runs/logs")


def train_cbow():
    hidden_size = 10
    losses = []
    loss_fn = nn.NLLLoss()
    model = CBOW(vocab_size, embd_size, CONTEXT_SIZE, hidden_size)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        total_loss = .0
        for i, data in enumerate(cbow_train_data):
            context, target = data
            ctx_idxs = [w2i[w] for w in context]
            ctx_var = torch.LongTensor(ctx_idxs)

            model.zero_grad()
            log_probs = model(ctx_var)

            loss = loss_fn(log_probs, torch.LongTensor([w2i[target]]))

            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        losses.append(total_loss / i)
        writer.add_scalar("Loss_epoch/cbow_train", total_loss / i, epoch + 1)
    return model, losses


def train_skipgram():
    losses = []
    loss_fn = nn.MSELoss()
    model = SkipGram(vocab_size, embd_size)
    # print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        total_loss = .0
        for i, data in enumerate(skipgram_train_data):
            in_w, out_w, target = data
            in_w_var = torch.LongTensor([w2i[in_w]])
            out_w_var = torch.LongTensor([w2i[out_w]])

            model.zero_grad()
            log_probs = model(in_w_var, out_w_var)
            loss = loss_fn(log_probs[0], torch.Tensor([target]))

            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        losses.append(total_loss)
        writer.add_scalar("Loss_epoch/skip_train", total_loss / i, epoch + 1)
    return model, losses


cbow_model, cbow_losses = train_cbow()
sg_model, sg_losses = train_skipgram()


# test
# You have to use other dataset for test, but in this case I use training data because this dataset is too small
def test_cbow(test_data, model):
    print('====Test CBOW===')
    correct_ct = 0
    for ctx, target in test_data:
        ctx_idxs = [w2i[w] for w in ctx]
        ctx_var = torch.LongTensor(ctx_idxs)

        model.zero_grad()
        log_probs = model(ctx_var)
        _, predicted = torch.max(log_probs.data, 1)
        predicted_word = i2w[predicted.item()]
        print('predicted:', predicted_word)
        print('label    :', target)
        if predicted_word == target:
            correct_ct += 1

    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct / len(test_data) * 100, correct_ct, len(test_data)))


def test_skipgram(test_data, model):
    print('====Test SkipGram===')
    correct_ct = 0
    for in_w, out_w, target in test_data:
        in_w_var = torch.LongTensor([w2i[in_w]])
        out_w_var = torch.LongTensor([w2i[out_w]])

        model.zero_grad()
        log_probs = model(in_w_var, out_w_var)
        _, predicted = torch.max(log_probs.data, 1)
        predicted = predicted[0]
        if predicted == target:
            correct_ct += 1

    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct / len(test_data) * 100, correct_ct, len(test_data)))


test_cbow(cbow_train_data, cbow_model)
print('------')
test_skipgram(skipgram_train_data, sg_model)

x = list(range(1, n_epoch + 1))
plt.plot(x, cbow_losses)
plt.show()
plt.plot(x, sg_losses)
plt.show()
