import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        batch_size = anchor.shape[0]
        positive_similarity = torch.sum(anchor * positive, dim=1, keepdim=True)
        negative_similarity = torch.matmul(anchor, negatives.transpose(1, 2))
        negative_similarity = negative_similarity.reshape(batch_size, -1)
        logits = torch.cat([positive_similarity, negative_similarity], dim=1)
        logits /= self.temperature
        softmax = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.log(softmax[:, 0] + 1e-10))
        return loss
