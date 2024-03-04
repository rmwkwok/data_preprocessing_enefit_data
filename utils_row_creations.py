'''Utilities for creating missing rows.'''

from pandas import (
    Index,
    MultiIndex,
    DataFrame,
    Timedelta,
    date_range,
)


def get_row_index(targets: DataFrame, cache: dict) -> DataFrame:
    '''With respect to `Data.targets`, what rows are expected in other
    dataframes in `Data`.

    Args:
        targets: pandas.DataFrame
        cache: dict
            To cache the requirement that can be used with
            `apply_row_index` on other dataframes.
    '''
    unq_lat = cache['lat_lon']['latitude'].unique()
    unq_lon = cache['lat_lon']['longitude'].unique()
    unq_hr = targets['hour'].unique()
    unq_county = targets['county'].unique()
    dt0 = targets['datetime'].min().replace(hour=0)
    for_wea_datetime = date_range(
        dt0 - Timedelta(21, unit='H'),
        dt0 + Timedelta(25, unit='H'),
        # dt0 + Timedelta(26, unit='H'),
        freq='H',
    )
    his_wea_datetime = date_range(
        dt0 - Timedelta(37, unit='H'),
        dt0 - Timedelta(14, unit='H'),
        freq='H',
    )

    cols = ['county', 'is_business', 'product_type',
            'is_consumption', 'hour']
    cache['require.rev_tar'] = MultiIndex.from_frame(
        targets[cols].drop_duplicates()
    )

    cols = ['county', 'is_business', 'product_type']
    cache['require.clients'] = MultiIndex.from_frame(
        targets[cols].drop_duplicates()
    )

    cache['require.ele_prc'] = Index(unq_hr, name='hour')
    cache['require.gas_prc'] = None

    cache['require.his_wea_latlon'] = MultiIndex.from_product(
        (unq_lat, unq_lon, his_wea_datetime),
        names=['latitude', 'longitude', 'datetime']
    )
    cache['require.his_wea_county'] = MultiIndex.from_product(
        (unq_county, unq_hr),
        names=['county', 'hour']
    )
    cache['require.for_wea_latlon'] = MultiIndex.from_product(
        (unq_lat, unq_lon, for_wea_datetime),
        names=['latitude', 'longitude', 'datetime']
    )
    cache['require.for_wea_county'] = MultiIndex.from_product(
        (unq_county, for_wea_datetime),
        names=['county', 'datetime']
    )
    return targets


def apply_row_index(df: DataFrame, index: MultiIndex | Index) -> DataFrame:
    '''Apply `index` obtained with `get_row_index`.

    Missing but required rows will be NaNs.
    '''
    return (
        df
        .set_index(index.names)
        .reindex(index)
        .reset_index()
    )

