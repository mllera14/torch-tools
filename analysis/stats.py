import numpy as np
import pandas as pd

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

from fitting import test, compute_accuracy

########################################################################################
# PCA FUNCTIONS
########################################################################################

def per_timestep_pca(data, n_components=None, random_state=None):
    grouped_by_time = data.groupby(level=['timestep'])
    units = data.index.unique(level='unit')
    inputs = data.index.unique(level='input')

    models, projections = [], []
    for name, group in grouped_by_time:
        X = group.values.reshape(-1, len(units))
        pca = PCA(
            n_components=n_components,
            copy=False, whiten=False,
            random_state=random_state
        ).fit(X=X)
        X_proj = pca.transform(X)

        models.append(pca)
        projections.append(X_proj)

    projections = pd.Series(
        data=np.asarray(projections).reshape(-1),
        index=pd.MultiIndex.from_product(
            [data.index.unique(level='timestep'), inputs, range(models[0].n_components_)],
            names=['timestep', 'input', 'component']
        ),
        name='activation projection'
    )

    return projections, models


def per_unit_pca(data, n_components=None, random_state=None):
    X = data.groupby(level=['timestep', 'unit']).mean().values
    X = X.reshape(-1, len(data.index.unique(level='unit')))

    pca = PCA(
        n_components=n_components,
        copy=False, whiten=False,
        random_state=random_state
    ).fit(X=X)

    X_proj = pca.transform(X)
    X_proj /= 3 * X_proj.std()
    # X_proj /= np.sqrt(pca.singular_values_[0])

    X_proj = pd.Series(
        data=X_proj.reshape(-1),
        index=pd.MultiIndex.from_product(
            [data.index.unique(level='timestep'), range(pca.n_components_)],
            names=['timestep', 'component']
        ),
        name='activation projection'
    )

    return X_proj, pca


def pca_to_dataframe(models):
    timesteps = range(len(models))
    components = range(models[0].n_components_)
    features = range(models[0].components_.shape[1])

    muidx = pd.MultiIndex.from_product(
            [timesteps, components],
            names=['timestep', 'component'])

    s = pd.Series(
        data=np.concatenate([m.explained_variance_.reshape(-1)  for m in models]),
        index=muidx,
        name='explained variance'
    )

    pca_results = pd.DataFrame(
        data=np.concatenate([m.components_.reshape(-1, len(features)) for m in models]),
        index=muidx,
        columns=['feature {}'.format(i + 1) for i in features]
    )

    pca_results[s.name] = s

    return pca_results


########################################################################################
# TSNE FUNCTIONS
########################################################################################

def tsne_projection(data, n_components=2, perplexity=30.0):
    last_tstep = data.groupby(['input', 'unit']).last()
    units = sorted(data.index.unique(level='unit'))
    inputs = sorted(data.index.unique(level='input'))

    X = last_tstep.values.reshape(-1, len(units))
    X_proj = TSNE(n_components, perplexity).fit_transform(X)

    X_proj = pd.DataFrame(
        data=X_proj,
        index=pd.Index(inputs, names='inputs'),
        columns=['component #{}'.format(i + 1) for i in range(n_components)]
    )

    return X_proj


########################################################################################
# Analysis
########################################################################################

def weighted_activity(model, recordings):
    df = recordings.groupby(['input', 'unit']).last()
    weights = model.linear.weight.detach().numpy().T.reshape(-1)

    gb_act = df.set_index('class', append=True).groupby(['unit', 'class'])

    def weigh(group):
        # name in the same order given in groupby
        unit, label = group.name
        return group * weights[label, unit]

    weighted_activity = gb_act.apply(weigh)

    return weighted_activity


def mean_weighted_activity(model, recordings):
    df = recordings.groupby(['input', 'unit']).last()
    df = df.set_index(
        'class', append=True).groupby(['unit','class']).mean()
    df.columns = ['mean activation']

    df['weight'] = model.linear.weight.detach().numpy().T.reshape(-1)

    wact = weighted_activity(model, recordings)

    df['mean weighted activation'] = wact.groupby(['unit', 'class']).mean()

    return df


def test_hidden_representation(model, loader, device='cpu'):
    with torch.no_grad():
        test_score = compute_accuracy(model, loader, device)

        trained_layer = model.linear

        out_features, in_features = model.linear.weight.shape
        readout_layer = torch.nn.Linear(in_features, out_features)

        readout_layer.bias.fill_(0)
        readout_layer.weight.fill_(1)
        readout_layer.weight *= trained_layer.weight.sign()

        model.linear = readout_layer

        readout_score = compute_accuracy(model, loader, device)

        model.linear = trained_layer

    return test_score, readout_score

def readout_test(models, data_loader):
    scores = [
        score for model_scores in [test_hidden_representation(model, data_loader)
            for model in models] for score in model_scores
    ]

    model_names = [m.type for m in models]
    output_layer_type = ['standard', 'unit readout']

    scores = pd.Series(
        data=scores,
        index=pd.MultiIndex.from_product(
            [model_names, output_layer_type], names=['model', 'projection']),
        name='accuracy'
    )

    return scores


# def ablate_unit(model, unit):
#     for w in model.parameters():


# def ablated_score(model, dataloader, unit):
#     model_name = model.type

#     with torch.no_grad():
#         rnn = model.rnn
