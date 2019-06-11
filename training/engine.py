import numpy as np
import csv

import torch
import torch.nn as nn

from ignite.engine import Events, Engine, _prepare_batch
from tqdm import tqdm

########################################################################################
# Training
########################################################################################

def _detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """

    if hidden_state is None:
        return None
    elif isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [_detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(_detach_hidden_state(h) for h in hidden_state)
    raise ValueError('Unrecognized hidden state type {}'.format(type(hidden_state)))


def create_rnn_trainer(model, optimizer, loss_fn, grad_clip=0, track_hidden=True, device=None,
                    non_blocking=False, prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _training_loop(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()

        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
        hidden = engine.state.hidden

        # Forward pass
        pred, hidden = model(inputs, hidden)

        loss = loss_fn((pred, hidden), targets)
        engine.state.hidden = hidden

        # Backwards
        loss.backward()

        # Optimize
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        return loss.item()

    # If reusing hidden states, detach them from the computation graph
    # of the previous batch. Usin the previous value may speed up training
    # but detaching is needed to avoid backprogating to the start of training.
    def _detach_wrapper(engine):
        if track_hidden:
            engine.state.hidden = _detach_hidden_state(engine.state.hidden)
        else:
            engine.state.hidden = None

    engine = Engine(_training_loop)
    engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'hidden', None))
    engine.add_event_handler(Events.ITERATION_STARTED, _detach_wrapper)

    return engine


def create_feedforward_trainer(model, optimizer, loss_fn, grad_clip=0, device=None,
                    non_blocking=False, prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _training_loop(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()

        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)

        # Forward pass
        pred = model(inputs)
        loss = loss_fn(pred, targets)

        # Backwards
        loss.backward()

        # Optimize
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        return loss.item()

    return Engine(_training_loop)


########################################################################################
# Testing
########################################################################################

def create_rnn_evaluator(model, metrics, device=None, hidden=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            pred, _ = model(inputs, hidden)

            return pred, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_feedforward_evaluator(model, metrics, device=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model(inputs)
            return output, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
