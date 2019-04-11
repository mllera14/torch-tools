import os
import numpy as np
import pandas as pd
import torch


def record_activations(model, data, device):
    raw_data = []

    # set up recording forward hook
    def acitvation_recorder(self, input, output):
        out, _ = output

        try:
            out = out.numpy()
        except TypeError:
            out = out.cpu().numpy()

        raw_data.append(out)

    hook = model.rnn.register_forward_hook(acitvation_recorder)

    # feed stimuli to network
    with torch.no_grad():
        for i, batch in enumerate(data):
            inputs, _ = batch
            inputs = inputs.to(device)

            outputs = model(inputs)[0]

    hook.remove()
    raw_data = np.concatenate(raw_data)

    # Transform data to Pandas DataFrame

    input_idx = range(raw_data.shape[0])
    timesteps = range(raw_data.shape[1])
    units =  range(raw_data.shape[2])

    s = pd.Series(
        data=raw_data.reshape(-1),
        index=pd.MultiIndex.from_product(
            [input_idx, timesteps, units],
            names=['input','timestep', 'unit']),
        name='activation')

    return s
