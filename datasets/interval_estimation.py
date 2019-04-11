import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def sample_data(batch_size, seq_len, rng):
    inputs = np.zeros((batch_size, seq_len, 1))

    # Neat trick to sample the positions to unmask
    mask = rng.rand(batch_size, seq_len).argsort(axis=1)[:, :2]
    mask.sort(axis=1)

    # Mask is in the wrong shape (batch_size, num_addends) for slicing
    inputs[range(batch_size), mask.T, 0] = 1

    # Get target values. Multiply each addend by its mask and sum over all of them
    nz = np.flatnonzero(inputs).reshape(-1, 2)
    targets = np.abs(nz[:, 0] - nz[:, 1])
    targets = targets.reshape(-1, 1)

    return inputs, targets


class BatchGenerator:
    def __init__(self, size=1000, seq_len=10, batch_size=20, random_state=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.size = size
        self._emitted = 0

        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        else:
            random_state = random_state

        self.rng = random_state
        self.init_state = random_state.get_state()

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

    def __len__(self):
        return self.size // self.batch_size

    def n_instances(self):
        return self.size * self.batch_size

    def reset(self):
        self.rng.set_state(self.init_state)
        self._emitted = 0

    def next_batch(self):
        if self._emitted > self.size:
            self.reset()

        inputs, targets = sample_data(
            batch_size=self.batch_size, seq_len=self.seq_len, rng=self.rng)

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        targets = torch.as_tensor(targets, dtype=torch.float)

        if self.size > 0:
            self._emitted += self.batch_size

        return inputs, targets

    def torch_dataset(self):
        current_state = self.rng.get_state()
        self.rng.set_state(self.init_state)

        inputs, targets = sample_data(
            batch_size=self.size, seq_len=self.seq_len, rng=self.rng)

        self.rng.set_state(current_state)

        data = TensorDataset(
            torch.as_tensor(inputs, dtype=torch.float),
            torch.as_tensor(targets, dtype=torch.float)
        )

        return DataLoader(
            dataset=data,
            batch_size=self.batch_size, shuffle=False
        )