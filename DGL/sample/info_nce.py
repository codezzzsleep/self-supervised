import torch
import torch.nn.functional as F


def infonce_loss(anchor, positive, negatives, temperature=0.1):
    """
    计算InfoNCE损失。

    参数：
        anchor (torch.Tensor): 形状为 (batch_size, feature_dim) 的锚点张量。
        positive (torch.Tensor): 形状为 (batch_size, feature_dim) 的正实例张量。
        negatives (torch.Tensor): 形状为 (batch_size, num_negatives, feature_dim) 的负实例张量。
        temperature (float): 温度参数，用于缩放 logits。
    返回：
        loss (torch.Tensor): InfoNCE损失。
    """
    batch_size = anchor.shape[0]

    # 计算锚点和正实例之间的相似度
    positive_similarity = torch.sum(anchor * positive, dim=1, keepdim=True)

    # 计算锚点和负实例之间的相似度
    negative_similarity = torch.matmul(anchor, negatives.transpose(1, 2))

    # 重塑负相似度张量使其在第 1 维度上具有与正相似度相同的尺寸
    negative_similarity = negative_similarity.reshape(batch_size, -1)

    # 进行拼接，形状为(batch_size, 1 + num_negatives)
    logits = torch.cat([positive_similarity, negative_similarity], dim=1)

    # 缩放 logits
    logits /= temperature

    # 计算 softmax 并获得损失值
    softmax = F.softmax(logits, dim=1)
    loss = -torch.mean(torch.log(softmax[:, 0] + 1e-10))

    return loss


anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negatives = torch.randn(32, 5, 128)  # 5个负实例

loss = infonce_loss(anchor, positive, negatives)

print(loss)
