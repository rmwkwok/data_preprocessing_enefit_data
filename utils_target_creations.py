from numpy.linalg import lstsq
from pandas import DataFrame, Series


EPSILON = 1e-7


def _lstsq(df: DataFrame, label: str):
    '''Returns least square solutions and a std value.'''
    X = df.drop(columns=label).astype('float')
    y = (df[label] + EPSILON).apply('log').astype('float')
    w = Series(lstsq(X.values, y.values, rcond=None)[0], index=X.columns)
    w['_std'] = (df[label] - (X @ w).apply('exp') + EPSILON).std()
    return w


def transform_target_1(
    df: DataFrame, label: str, new_target_name: str,
    cache: dict, training: bool=True, reverse: bool=False,
) -> DataFrame:
    cache = cache.setdefault(f'transform_target_1_{new_target_name}', {})

    y = df[label]
    f1 = df['is_consumption']
    f2 = df['installed_capacity']
    factor = f1.replace(0, nan).fillna(f2)
    if reverse:
        new_target = y * factor
    else:
        new_target = y / factor
    return df.assign(**{new_target_name: new_target})


def transform_target_2(
    df: DataFrame, label: str, new_target_name: str,
    cache: dict, training: bool=True, reverse: bool=False,
) -> DataFrame:
    cache = cache.setdefault(f'transform_target_2_{new_target_name}', {})

    indp = ['county', 'product_type']
    cond = ['is_business', 'is_consumption', 'month', 'weekday', 'hour']

    _df = df.set_index(cond)[indp + [label]].assign(ones=1.)
    _df = get_dummies(_df, columns=indp)

    if training:
        W = _df.groupby(cond).apply(_lstsq, label=label)
        cache['W'] = W.drop(columns='_std')
        cache['std'] = W['_std'] + EPSILON

    X = _df.drop(columns=label)
    y = _df[label]
    W = cache['W'].reindex(_df.index)
    s = cache['std'].reindex(_df.index)
    offset = (X * W).sum(axis=1).apply('exp') - EPSILON

    if reverse:
        new_target = (y * s + offset)
    else:
        new_target = ((y - offset) / s)
    new_target.index = df.index
    return df.assign(**{new_target_name: new_target})


def transform_target_3(
    df: DataFrame, label: str, new_target_name: str,
    cache: dict, training: bool=True, reverse: bool=False,
) -> DataFrame:
    cache = cache.setdefault(f'transform_target_3_{new_target_name}', {})

    steps = [
        transform_target_1,
        transform_target_2,
    ]
    if reverse:
        steps = reversed(stes)

    for step in steps:
        df = step(df, label, new_target_name, cache, training, reverse)
        label = new_target_name
    return df

