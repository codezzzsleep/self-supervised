import torch


def save_model(model, path):
    torch.save(model.state_dict(), 'save_model/best_model.pth')
