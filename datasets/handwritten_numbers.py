import numpy as np
from itertools import product, cycle
import torch

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as trans
import torchvision.datasets as dataset


def group_img_by_class(ixc_pair):
    ixc_pair  = sorted(ixc_pair, key=lambda x: x[0])

    current = ixc_pair[0][0]
    groups = {current: []}

    for num, idx in ixc_pair:
        if num == current:
            groups[current].append(idx)
        else:
            current = num
            groups[current] = [idx]

    return groups


class RandomGenerator:
    def __init__(self, nsamples, ndigits, groups, rng=None, filter_fn=None):
        self.ndigits = ndigits
        self.idx = groups
        self.classes = list(groups.keys())
        self.rng = rng if rng is not None else np.random
        self.nsamples = nsamples
        self.filter = filter_fn

    @property
    def n_classes(self):
        return len(self.classes)

    def _generate(self, all_numbers):
        digits = all_numbers[self.rng.choice(len(all_numbers))]
        # digits = self.rng.choice(list(all_numbers))
        idx = [self.rng.choice(self.idx[d]) for d in digits]

        return idx[::-1]

    def __iter__(self):
        init_state = self.rng.get_state()
        all_numbers = list(filter(self.filter, product(self.classes, repeat=self.ndigits)))

        for _ in range(self.nsamples):
            yield self._generate(all_numbers)

        self.rng.set_state(init_state)


class OrderedGenerator:
    def __init__(self, nsamples, ndigits, groups, rng=None, filter_fn=None):
        self.ndigits = ndigits
        self.idx = groups
        self.classes = list(groups.keys())
        self.rng = np.random
        self.nsamples = nsamples
        self.filter = filter_fn

    def _generate(self, digits):
        idx = [self.rng.choice(self.idx[d]) for d in digits]
        return idx

    def __iter__(self):
        init_state = self.rng.get_state()

        all_numbers = filter(
            self.filter, product(self.classes, repeat=self.ndigits))

        for i, digits in enumerate(cycle(all_numbers)):
            if i >= self.nsamples:
                break
            else:
                yield self._generate(list(digits))

        self.rng.set_state(init_state)


class HWND(Dataset):
    def __init__(self, mnist, ndigits, shape, dataset_size,
                    rng=None, sampler='random', filter_fn=None):
        self.mnist = mnist
        self.ndigits = ndigits
        self.dataset_size = dataset_size
        self.rng = rng
        self.shape = shape

        ixc_pair = [(mnist[i][1], i) for i in range(len(mnist))]
        groups = group_img_by_class(ixc_pair)

        if type(sampler) == str:
            if sampler == 'random':
                sampler = RandomGenerator(
                    dataset_size, ndigits, groups, self.rng, filter_fn)
            elif sampler == 'ordered':
                sampler = OrderedGenerator(
                    dataset_size, ndigits, groups, self.rng, filter_fn)
            else:
                raise ValueError('Unknown sampling procedure')

        indexes = []

        for i, imgidx in enumerate(sampler):
            if i >= self.dataset_size:
                break
            self._trim_leading_zeros(imgidx)
            indexes.append(imgidx)

        self.empty_image = torch.zeros_like(mnist[0][0])
        self.img_idx = indexes

    def _trim_leading_zeros(self, imgidx):
        for i in range(len(imgidx) - 1):
            if self.mnist[imgidx[i]][1] == 0:
                imgidx[i] = -1
            else:
                break

    def _get_target(self, idx):
        digits = [self.mnist[d][1] for d in self.img_idx[idx]]
        label = 0
        for i, d in enumerate(digits[::-1]):
            if d == -1:
                d = 0
            label += d * (10 ** i)
        return label

    def _get_images(self, idx):
        images = []
        for i in self.img_idx[idx]:
            if i == -1:
                images.append(self.empty_image.view(*self.shape))
            else:
                images.append(self.mnist[i][0].view(*self.shape))

        return torch.cat(images, dim=1).view(1, -1)


    def __getitem__(self, index):
        images = self._get_images(index)
        label = self._get_target(index)

        return images, label

    def __len__(self):
        return len(self.img_idx)


class HWNSeq(Dataset):
    def __init__(self, generators):
        self.generators = generators
        self.seqlen = len(generators)

        indexes = product(*[range(len(g)) for g in generators])
        self.indexes = list(indexes)

    def __getitem__(self, idx):
        images, labels = [], []

        indexes = self.indexes[idx]

        for i, g in zip(indexes, self.generators):
            img, label = g[i]
            images.extend(img)
            labels.append(label)

        images = torch.stack(images)
        labels = torch.as_tensor(labels)

        return images, labels


    def __len__(self):
        return len(self.indexes)


def load_hwnd(mnist_path, dataset_size, ndigits=3, crop_shape=(22, 20), batch_size=50,
                val_split_ratio=0.2, rng=None, download=False):
    train_raw, test_raw = load_raw(mnist_path, crop_shape, download)

    training_set = HWND(train_raw, ndigits, crop_shape, dataset_size, rng)
    test_set = HWND(test_raw, ndigits, crop_shape, dataset_size, rng)

    # Split train data into training and validation sets
    if val_split_ratio > 0:
        N = len(training_set)
        val_size = int(N * val_split_ratio)
        training_set, validation_set = torch.utils.data.random_split(
            training_set, [N - val_size, val_size])
        validation_set = DataLoader(validation_set, batch_size=val_size, shuffle=False)
    else:
        validation_set = None

    training_set = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=len(train_raw), shuffle=False)

    return training_set, test_set, validation_set


def load_hwnseq(
    mnist_path,
    seqlen, examples_per_pos,
    ndigits, digit_shape,
    train_filters=None,
    test_filters=None,
    batch_size=50,
    val_split_ratio=0.2,
    rng=None,
    download=False):

    train_raw, test_raw = load_raw(mnist_path, digit_shape, download)

    training_gen = []
    test_gen = []

    for i in range(seqlen):
        filter_fn = None if train_filters is None else train_filters[i]
        tr_set = HWND(train_raw, ndigits, digit_shape, examples_per_pos, filter_fn=filter_fn, rng=rng)

        filter_fn = None if test_filters is None else test_filters[i]
        te_set = HWND(test_raw, ndigits, digit_shape, examples_per_pos, filter_fn=filter_fn, rng=rng)

        training_gen.append(tr_set)
        test_gen.append(te_set)

    training_set = HWNSeq(training_gen)
    test_set = HWNSeq(test_gen)

    if val_split_ratio > 0:
        N = len(training_set)
        val_size = int(N * val_split_ratio)
        training_set, validation_set = torch.utils.data.random_split(
            training_set, [N - val_size, val_size])
        validation_set = DataLoader(validation_set, batch_size=val_size, shuffle=False)
    else:
        validation_set = None

    training_set = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    return training_set, test_set, validation_set


def load_raw(data_path, crop_shape=(22, 20), download=False):
    transform = trans.Compose([
        trans.CenterCrop(crop_shape),
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(-1, np.prod(x.shape)))
    ])

    train_data = dataset.MNIST(
        root=data_path, train=True, transform=transform, download=download)
    test_data = dataset.MNIST(
        root=data_path, train=False, transform=transform, download=download)

    return train_data, test_data
