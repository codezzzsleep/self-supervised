from torch.utils.data import DataLoader
from dataset import SkipGramDataset


def load_data(text=None, window_size=None, batch_size=None):
    if text is None:
        text = "this is a simple example for word2vec using pytorch"
    if window_size is None:
        window_size = 2
    if batch_size is None:
        batch_size = 4
    dataset = SkipGramDataset(text, window_size=window_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, dataset
