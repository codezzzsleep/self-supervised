import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import utils
from net import Word2Vec
from train import train

writer = SummaryWriter("runs/logs")

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

dataset, data_loader = utils.load_data(text)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(dataset.word_to_idx)
model = Word2Vec(vocab_size, embed_size)
model.to(device)

# 训练模型
loss_list = train(model, data_loader, epochs, learning_rate, device, writer)
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
