from itertools import product, chain


def model_folder(params, seed):
    model_type = params['model']
    nlayers = params['nlayers']
    nhid = params['nhid']

    return  '/{}-l{}-h{}-s{}'.format(model_type, nlayers, nhid, seed)


def get_cmd(args):
    args_list = ['--{} {}'.format(k, v) for (k, v) in args.items()]

    return ' '.join(args_list)


def get_sweep_mode(mode):
    if mode == '1to1':
        return zip
    elif mode == 'allcomb':
        return product
    else:
        raise ValueError('Unidentified parameter combination mode {}'.format(mode))


def param_sweep_iter(*args):
    fixed_params, sweep_values, sweep_keys = {}, [], []
    comb_mode = None

    for k, v in chain(*args):
        if k == 'sweep-mode':
            comb_mode = get_sweep_mode(v)
        elif isinstance(v, list):
            sweep_values.append(v)
            sweep_keys.append(k)
        else:
            fixed_params[k] = v

    if comb_mode is None:
        comb_mode = product

    for comb in comb_mode(*sweep_values):
        sweep_params = dict(zip(sweep_keys, comb))
        params = {**fixed_params, **sweep_params}

        yield params

