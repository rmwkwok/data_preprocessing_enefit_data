'''Utilities for missing values imputations.'''

from typing import Optional


import warnings
from pandas import (
    DataFrame,
    concat,
)
from pandas.errors import PerformanceWarning


def nearestfill(
    df: DataFrame,
    *,
    sort_columns: Optional[list[str] | str]=None,
    group_columns: Optional[list[str] | str]=None,
    fill_columns: Optional[list[str] | str]=None,
) -> DataFrame:
    '''Fill NaNs in numeric columns with averaged nearest values.

    Args:
        df: pandas.DataFrame
        sort_columns: Optional[list[str] | str]=None
            Columns to sort on. This affect what values are
            nearest to the NaNs.
        group_columns: Optional[list[str] | str]=None
            Columns to group on. Only in-group values can be
            considered nearest.
        fill_columns: Optional[list[str] | str]=None
            Numeric columns to fill. If `None`, all numeric
            columns except for `group_columns`.
    '''
    def _type_check(columns):
        return [columns, ] if isinstance(columns, str) else columns

    sort_columns = _type_check(sort_columns)
    group_columns = _type_check(group_columns)
    fill_columns = _type_check(fill_columns)

    if sort_columns:
        tmp = df.sort_values(sort_columns)
    else:
        tmp = df

    tmp = tmp.select_dtypes('number')

    if group_columns:
        tmp = tmp.groupby(group_columns)
    else:
        tmp = tmp

    if fill_columns:
        tmp = tmp[fill_columns]

    ffill = tmp.ffill()
    bfill = tmp.bfill()
    return df.fillna((ffill.fillna(bfill) + bfill.fillna(ffill)) / 2)


def fill_missing_clients(df: DataFrame, cache: dict) -> DataFrame:
    '''Fill the clients dataframe with any missing historic clients. For
    new clients required by `Data.targets`, fill with default data.

    Args:
        df: pandas.DataFrame
        cache: dict
            Cache to store the latest full clients list.
    '''
    df = df.set_index(['county', 'is_business', 'product_type'])

    name = 'all_clients_list'
    cached = cache.setdefault(name, df.copy())

    missing_clients = cached.index.difference(df.index)
    df = concat([df, cached.loc[missing_clients]])
    cache[name] = df.copy()

    # Default values are based on these new clients at their early days.
    # unit_id, is_business, product_type, eic_count, installed_capacity
    # 61, 1, 2, 6, 16,
    # 62, 0, 1, 6, 83,
    # 63, 1, 1, 5, 185,
    # 64, 1, 0, 8, 260
    # 65, 1, 1, 5, 95,
    # 66, 1, 1, 8, 462.6,
    # 67, 1, 0, 7, 280,
    # 68, 1, 2, 5, 149.5
    # median= 6, 168
    df['eic_count'] = df['eic_count'].fillna(6.)
    df['installed_capacity'] = df['installed_capacity'].fillna(168.)
    return df.reset_index()


def copy_county_6_data_to_12(df: DataFrame, group_columns: str) -> DataFrame:
    '''Consider county 12 as county 6, so that latitude and longitude
    data can be applied to them.'''
    return df.fillna(
        df[df['county'].isin([6, 12])]
        .sort_values('county')
        .groupby(group_columns)
        .ffill()
    )


def replace_strange_latlon(df: DataFrame) -> DataFrame:
    '''Replace the following two regions of historical weather data:
    - replace (57.6, 23.2) with (57.9, 23.2)
    - replace (57.6, 24.2) with (57.9, 24.2)
    '''
    index = ['latitude', 'longitude', 'datetime']
    if 'data_block_id' in df:
        index = index + ['data_block_id']
    df = df.set_index(index)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PerformanceWarning)
        df.loc[(576, 232)].update(df.loc[(579, 232)])
        df.loc[(576, 242)].update(df.loc[(579, 242)])
    return df.reset_index()

