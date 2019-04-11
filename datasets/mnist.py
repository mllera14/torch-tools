from torch.utils.data import DataLoader, random_split
import torchvision.transforms as trans
import torchvision.datasets as dataset


def load_raw(data_path, input_size, download=False, transform=None):
    transform = transform or trans.Compose([
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(-1, input_size))
    ])

    train_data = dataset.MNIST(
        root=data_path, train=True, transform=transform, download=download)
    test_data = dataset.MNIST(
        root=data_path, train=False, transform=transform, download=download)

    return train_data, test_data


def load_mnist(data_path, input_size, batch_size, val_split_ratio, shuffle=True, download=False):
    train_raw, test_raw = load_raw(data_path, input_size, download)

    # Split train data into training and validation sets
    N = len(train_raw)
    val_size = int(N * val_split_ratio)
    train_raw, validation_raw = random_split(
        train_raw, [N - val_size, val_size])

    train_data = DataLoader(train_raw, batch_size=batch_size, shuffle=shuffle)
    validation_data = DataLoader(validation_raw, batch_size=val_size, shuffle=False)
    test_data = DataLoader(test_raw, batch_size=len(test_raw), shuffle=False)

    return train_data, validation_data, test_data