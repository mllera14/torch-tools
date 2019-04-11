import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def sample_data(
    batch_size,
    seq_len,
    min_value,
    max_value,
    num_values,
    noise_var,
    valtype='int',
    rng=None
):

    # Neat trick to sample the positions to unmask
    if valtype == 'int':
        rand = rng.random_integers
    elif valtype == 'float':
        rand = rng.uniform
    else:
        raise ValueError()

    inputs = rand(
        low=min_value, high=max_value, size=(batch_size, seq_len, 2))
    inputs[:, :, 1] = 0

    mask = rng.randn(batch_size, seq_len).argsort(axis=1)[:,:num_values]
    mask.sort(axis=1)

    # Mask is in the wrong shape (batch_size, num_addends) for slicing
    inputs[range(batch_size), mask.T, 1] = 1

    # Get target values. Multiply each addend by its mask and sum over all of them
    targets = np.sum(inputs[:, :, 0] * inputs[:, :, 1], axis=1).reshape(-1, 1)

    if noise_var > 0:
        zeta = rng.randn(batch_size, seq_len)
        inputs[:, :, 0] += zeta * np.sqrt(noise_var)
        inputs[:, :, 0] = inputs[:, :, 0].clip(min_value, max_value)

    return inputs, targets

def get_mask_labels(dataset):
    ops = dataset.swapaxes(1, 2).swapaxes(0, 1)[1]
    labels = np.asarray(['add' if i else 'ignore' for i in ops.reshape(-1)])
    return labels.reshape(ops.shape)


class BatchGenerator:
    def __init__(self, size=1000, seq_len=10, num_values=2, noise_var=0.0,
            min_value=0, max_value=1, batch_size=10, random_state=None):
        self.min_value = min_value
        self.max_value = max_value
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.size = size
        self.num_values = num_values
        self.noise_var = noise_var
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
            batch_size=self.batch_size, seq_len=self.seq_len,
            min_value=self.min_value, max_value=self.max_value,
            noise_var=self.noise_var, num_values=self.num_values,
            valtype='float', rng=self.rng
        )

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        targets = torch.as_tensor(targets, dtype=torch.float)

        if self.size > 0:
            self._emitted += self.batch_size

        return inputs, targets

    def torch_dataset(self):
        current_state = self.rng.get_state()
        self.rng.set_state(self.init_state)

        inputs, targets = sample_data(
            batch_size=self.batch_size, seq_len=self.seq_len,
            min_value=self.min_value, max_value=self.max_value,
            noise_var=self.noise_var, num_values=self.num_values,
            valtype='float', rng=self.rng
        )

        self.rng.set_state(current_state)

        data = TensorDataset(
            torch.as_tensor(inputs, dtype=torch.float),
            torch.as_tensor(targets, dtype=torch.float)
        )

        return DataLoader(
            dataset=data,
            batch_size=self.batch_size, shuffle=False
        )


def load_delayed_addition(
    N,
    batch_size,
    test_size,
    seq_len,
    num_addends,
    min_addend,
    max_addend,
    train_val_split,
    train_noise_var,
    test_noise_var,
    rng=None
):
    train_size = int(N * (1 - train_val_split))
    val_size = N - train_size

    if rng is None:
        data_rng = np.random.RandomState(np.random.randint(2**32-1))
        test_rng = np.random.randint(2**32-1)

    training_data = BatchGenerator(
        size=train_size, seq_len=seq_len, num_values=num_addends,
        min_value=min_addend, max_value=max_addend, batch_size=batch_size,
        noise_var=train_noise_var, random_state=data_rng)

    validation_data = BatchGenerator(
        size=val_size, seq_len=seq_len, num_values=num_addends,
        min_value=min_addend, max_value=max_addend, batch_size=val_size,
        noise_var=train_noise_var, random_state=data_rng)

    test_data = BatchGenerator(
        size=test_size, seq_len=seq_len, num_values=num_addends,
        min_value=min_addend, max_value=max_addend, batch_size=test_size,
        noise_var=test_noise_var, random_state=test_rng)

    return training_data, validation_data, test_data
