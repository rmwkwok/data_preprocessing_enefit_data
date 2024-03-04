'''Utilities for data readout and others.'''

from numpy import nan, inf, ndarray, isnan, isinf
from pandas import DataFrame, Series, read_csv
from pandas.api.types import is_numeric_dtype
from data_class import Data


TRAIN_DATA_DIR = './predict-energy-behavior-of-prosumers'


def flatten_column_index(df: DataFrame) -> DataFrame:
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def stat(data) -> str:
    if isinstance(data, Series):
        data = data.to_frame()
    if isinstance(data, DataFrame):
        data = data.replace(inf, nan).replace(-inf, nan)
        nan_inf = data.isna().sum().sum()
        non_num = sum(not is_numeric_dtype(data[c]) for c in data)
    elif isinstance(data, ndarray):
        nan_inf = isnan(data).sum() + isinf(data).sum()
        non_num = int(not is_numeric_dtype(data))
    else:
        return 'Not a DataFrame nor a Series nor a NDArray'
    return (
        f'NaN/Inf = \033[93m{nan_inf}\033[0m, '
        f'NonNum Cols = \033[93m{non_num}\033[0m, '
        f'shape = {data.shape}'
    )


def read_data() -> list[Data]:
    group_column = 'data_block_id'
    dataframes = dict(
        targets=read_csv(f'{TRAIN_DATA_DIR}/train.csv'),
        clients=read_csv(f'{TRAIN_DATA_DIR}/client.csv'),
        his_wea=read_csv(f'{TRAIN_DATA_DIR}/historical_weather.csv'),
        for_wea=read_csv(f'{TRAIN_DATA_DIR}/forecast_weather.csv'),
        ele_prc=read_csv(f'{TRAIN_DATA_DIR}/electricity_prices.csv'),
        gas_prc=read_csv(f'{TRAIN_DATA_DIR}/gas_prices.csv'),
        # sam_pre=None,
    )
    dataframes['rev_tar'] = dataframes['targets'].copy()
    dataframes['rev_tar']['data_block_id'] += 2

    groups = {}
    for df_name, dataframe in dataframes.items():
        for group_id, sub_df in dataframe.groupby(group_column):
            # filter off incomplete groups
            if group_id not in [0, 1, 638, 639]:
                setattr(
                    groups.setdefault(group_id, Data()),
                    df_name,
                    sub_df.drop(columns=group_column)
                )
    return list(groups.values())


def read_latlon() -> DataFrame:
    return read_csv(
        f'{TRAIN_DATA_DIR}/weather_station_to_county_mapping.csv'
    )