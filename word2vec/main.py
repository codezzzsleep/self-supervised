import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


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
        self.out_embedding = nn.Linear(embed_size, vocab_size)

    def forward(self, center_idx):
        center_embed = self.in_embedding(center_idx)
        scores = self.out_embedding(center_embed)
        return scores


loss_list = []
writer = SummaryWriter("runs/logs")

min_loss = 10000.0


# 训练函数
def train(model, data_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        run_loss = 0.0
        for batch, (center_idx, context_idx) in enumerate(data_loader):
            center_idx = center_idx.to(device)
            context_idx = context_idx.to(device)

            # 修改这一行。将上下文索引移除，输入仅包含中心词索引。
            scores = model(center_idx)
            loss = criterion(scores, context_idx)

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


# 示例文本
# text = "this is a simple example for word2vec using pytorch"
file_path = 'data/text8.train.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取文件内容
    text = file.read()

# 超参数设置
window_size = 2
embed_size = 50
batch_size = 4
epochs = 1
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
print("train done!")
torch.save(model.state_dict(), 'runs/last_model.pt')
writer.close()
x = list(range(1, epochs + 1))
plt.plot(x, loss_list)

# 添加标题和坐标轴标签
plt.title('loss/epoch')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# 显示图形
plt.show()
print("task done!")
