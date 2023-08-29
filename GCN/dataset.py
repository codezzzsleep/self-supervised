from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_data():
    print("load data……")
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    print(data)
    return data


def load_dataset():
    print("load dataset……")
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    return dataset


if __name__ == "__main__":
    data = load_data()
    dataset = load_dataset()
    print(dataset)
    print(len(dataset))
    print(dataset.num_features)
    print(dataset.num_classes)
