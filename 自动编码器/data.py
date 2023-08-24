from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)
train_dataset = datasets.MNIST(root="./data",
                               train=True,
                               transform=transform,
                               download=True)

train_dataloader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True)


def load_data():
    return train_dataloader
