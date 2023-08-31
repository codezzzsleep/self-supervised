import os
import time
from pathlib import Path

import numpy as np
import torch


def save_model(model, path):
    torch.save(model.state_dict(), f'{path}/best_model.pth')


# def seem_seed(seed):
def create_result_folder():
    # 创建结果文件夹
    result_folder = f"{time.strftime('%Y%m%d_%H%M%S')}"
    result_folder = os.path.join("../runs", result_folder)
    train_folder = os.path.join(result_folder, 'train')
    test_folder = os.path.join(result_folder, 'test')
    log_folder = os.path.join(result_folder, 'logs')
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)
    base_path = os.path.join("../runs")
    return train_folder, test_folder, log_folder, base_path


# 示例：将一些结果保存到结果文件夹中
def save_result(model, result_folder, is_train=True):
    if is_train:
        folder = result_folder[0]  # 使用训练集文件夹路径
    else:
        folder = result_folder[1]  # 使用测试集文件夹路径
    torch.save(model.state_dict(), f'{folder}/last_model.pt')
    print(f"结果已保存到文件夹：{folder}")


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_best_model(model, loss, best_loss, best_model_path):
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), f'{best_model_path}/best_model.pt')
    return best_loss

# def draw_loss_graph(x,y,color,label)
