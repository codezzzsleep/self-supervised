import argparse
from tensorboard import program

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import load_dataset
from net import MyNet
from train import train
from test import test
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=800,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
# 配置
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')
# 数据导入
dataset = load_dataset()
data = dataset[0].to(device=device)
# 模型初始化
model = MyNet(num_feature=dataset.num_features, num_hidden=args.hidden,
              num_classes=dataset.num_classes, dropout=args.dropout).to(device)
# 优化器
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# 训练保存的路径
path = utils.create_result_folder()

# 启用tensorboard进行绘图
writer = SummaryWriter(path[2])

# 进行随机数种子设置
utils.same_seed(args.seed)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', path[3]])

# 启动 TensorBoard
url = tb.launch()

# 打印 TensorBoard 的 URL
print("TensorBoard启动成功！请访问下面的链接：")
print(url)
# 开始训练
train(model, data, args.epochs, optimizer, path,
      fastmode=args.fastmode)
# 测试
test(model, data)
print("Done!")
