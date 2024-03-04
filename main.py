'''Data preprocessing script.'''

import pickle
from tqdm import tqdm
from numpy import concatenate, squeeze, stack, newaxis, float32
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series, get_dummies

from utils_misc import *
from utils_row_creations import *
from utils_column_creations import *
from utils_target_creations import *
from utils_missing_values_imputations import *

NUM_LAT = 8 # Number of latitudes
NUM_LON = 14 # Number of longitudes
NUM_HOUR = 24 # Number of hours
FORECAST_COLUMNS = [
    'direct_solar_radiation', 'temperature', 'dewpoint',
    'cloudcover_total', 'cloudcover_low', 'cloudcover_mid',
    # 'snowfall', 'rain',
    'total_precipitation',
    'cloudcover_high',
    '10_metre_u_wind_component', '10_metre_v_wind_component',
    # 'windspeed_10m', 'cos_winddirection_10m', 'sin_winddirection_10m',
    'cos_month', 'sin_month',
    # 'cos_weekday', 'sin_weekday',
    'cos_lunarday', 'sin_lunarday',
    'cos_hour', 'sin_hour',
]
HISTORY_COLUMNS = [
    'surface_pressure', 'shortwave_radiation',  'diffuse_radiation',
    'direct_solar_radiation',
    # 'temperature', 'dewpoint',
    'cloudcover_total', 'cloudcover_low', 'cloudcover_mid',
]

### Data Readout
########################################################################
########################################################################

# The data is read in as a list of data blocks, where each block contains one
# day of data. Each block is represented by a `Data` class that contains a few
# DataFrames.
# The data may be downloaded at https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers
data_list = read_data()
lat_lon = read_latlon()
print('Data Readout Done.')

### Data Preprocessing
########################################################################
########################################################################

# A cache is used to pass variables created in processing one DataFrame to
# be used by the processing of another DataFrame.
cache = {
    'lat_lon': (
        lat_lon
        .drop(columns='county_name')
        .pipe(convert_latlon)
    )
}

#Each block of data is preprocessed separately over a loop.
print('Preprocessing begins.')
for data in tqdm(data_list):

    # Modifications to the dataframes one by one.
    data.targets = (
        data.targets
        .drop(columns=['prediction_unit_id'])
        .pipe(convert_datetime, ['datetime', 'prediction_datetime'])
        .pipe(
            datetime_split, 'datetime', encode=False,
            parts=['month', 'weekday', 'day', 'lunarday', 'hour'],
        )
        .pipe(
            datetime_split, 'datetime', encode=True,
            parts=['month', 'weekday', 'day', 'lunarday', 'hour'],
        )
        .pipe(add_holiday, 'datetime')
        .pipe(get_row_index, cache)
    )
    data.rev_tar = (
        data.rev_tar
        .drop(columns=['row_id', 'prediction_unit_id'])
        .rename(columns={'target': 'rev_target'})
        .pipe(convert_datetime, ['datetime'])
        .pipe(datetime_split, 'datetime', 'hour')
        .drop(columns=['datetime'])
        .pipe(apply_row_index, cache['require.rev_tar'])
    )
    data.clients = (
        data.clients
        .drop(columns=['date'])
        .pipe(apply_row_index, cache['require.clients'])
        .pipe(fill_missing_clients, cache.setdefault('clients', {}))
    )
    data.ele_prc = (
        data.ele_prc
        .drop(columns=['origin_date'])
        .pipe(convert_datetime, ['forecast_date'])
        .pipe(datetime_split, 'forecast_date', 'hour')
        .pipe(apply_row_index, cache['require.ele_prc'])
        # Two missing hours: '2022-03-28 02:00:00', '2023-03-27 02:00:00'
        .pipe(nearestfill, fill_columns='euros_per_mwh')
        .drop(columns=['forecast_date'])
    )
    data.gas_prc = (
        data.gas_prc
        .drop(columns=['origin_date', 'forecast_date'])
    )
    data.his_wea = (
        data.his_wea
        .pipe(unify_weather_data, is_history=True)
        .pipe(convert_latlon)
        .pipe(convert_datetime, ['datetime'])
        # Two repeated datetime at 2022-11-12 15:00:00
        .drop_duplicates(['datetime', 'latitude', 'longitude'])
        # .pipe(replace_strange_latlon)
        .pipe(apply_row_index, cache['require.his_wea_latlon'])
    )
    data.for_wea = (
        data.for_wea
        .drop(columns='origin_datetime')
        .rename(columns={'forecast_datetime': 'datetime'})
        .pipe(convert_datetime, ['datetime'])
        .pipe(unify_weather_data, is_history=False)
        .pipe(convert_latlon)
        .drop_duplicates(['datetime', 'latitude', 'longitude'])
        .pipe(apply_row_index, cache['require.for_wea_latlon'])
        .pipe(nearestfill, group_columns=['latitude', 'longitude'])
    )

    # Create temporary dataframes that are used to produce `x1`, `x2`, `x3`
    # and `y` which are added to the data block and for neural network
    # training.
    his_wea_tmp = (
        data.his_wea
        .pipe(
            datetime_split, 'datetime',
            ['month', 'weekday', 'lunarday', 'hour'], encode=True
        )
        .set_index(['datetime', 'latitude', 'longitude'])
        .sort_index()
        .reindex(columns=HISTORY_COLUMNS)
    )
    for_wea_tmp = (
        data.for_wea
        .pipe(
            datetime_split, 'datetime',
            ['month', 'weekday', 'lunarday', 'hour'], encode=True
        )
        .set_index(['datetime', 'latitude', 'longitude'])
        .sort_index()
        .reindex(columns=FORECAST_COLUMNS)
    )
    tmp_a = for_wea_tmp.index.get_level_values('datetime')
    tmp_b = tmp_a - tmp_a.min()
    for_wea_hour = tmp_b.days * 24 + tmp_b.seconds // 3600
    his_wea_hour = his_wea_tmp.index.get_level_values('datetime').hour
    x1 = (
        for_wea_tmp[(
            (for_wea_hour >= 19) & (for_wea_hour <= 46)
        )]
        .values
        .reshape(46 - 19 + 1, NUM_LAT, NUM_LON, len(FORECAST_COLUMNS))
    )
    x2 = (
        for_wea_tmp[(
            (for_wea_hour >= 0) & (for_wea_hour <= 9)
        )]
        .values
        .reshape( 9 -  0 + 1, NUM_LAT, NUM_LON, len(FORECAST_COLUMNS))
    )
    data.x1 = squeeze(
        sliding_window_view(
            x1,
            window_shape=(5, NUM_LAT, NUM_LON, len(FORECAST_COLUMNS)),
        )
    ).astype(float32)
    data.x2 = squeeze(
        sliding_window_view(
            x2,
            window_shape=(5, NUM_LAT, NUM_LON, len(FORECAST_COLUMNS)),
        )
    ).astype(float32)
    data.x3 = (
        his_wea_tmp[(
            ((his_wea_hour >= 5) & (his_wea_hour <= 10))
        )]
        .values
        .reshape(10 -  5 + 1, NUM_LAT, NUM_LON, len(HISTORY_COLUMNS))
        .astype(float32)
    )
    data.y = (
        his_wea_tmp
        .values
        .reshape(24, NUM_LAT, NUM_LON, len(HISTORY_COLUMNS))
        .astype(float32)
    )

    # Create new dataframe that merges all dataframes for a tabulated datasaet
    # used in training a Gradient-Boosted Decision Tree model.
    data.his_wea_tee_1 = (
        data.his_wea
        # Two versions of preprocessed history weather are teed
        # out from this point. The first version is for XGB,
        # whereas the second version is for NN training.
        # History Weather Version 1
        .pipe(datetime_split, 'datetime', 'hour')
        .merge(cache['lat_lon'], how='left', on=['latitude', 'longitude'])
        .drop(columns=['latitude', 'longitude', 'datetime'])
        .groupby(['hour', 'county'])
        .mean()
        .reset_index()
        .pipe(apply_row_index, cache['require.his_wea_county'])
        .pipe(copy_county_6_data_to_12, group_columns=['hour'])
    )
    data.for_wea_tee_1 = (
        data.for_wea
        .merge(cache['lat_lon'], how='left', on=['latitude', 'longitude'])
        .drop(columns=['latitude', 'longitude', 'hours_ahead'])
        .groupby(['datetime', 'county'])
        .mean()
        .reset_index()
        .pipe(apply_row_index, cache['require.for_wea_county'])
        .pipe(copy_county_6_data_to_12, group_columns=['datetime'])
    )
    data.merged = (
        data.targets
        .merge(
            data.rev_tar,
            how='left', on=['county', 'is_business', 'product_type', 'is_consumption', 'hour']
        )
        .merge(
            data.clients,
            how='left', on=['product_type', 'county', 'is_business']
        )
        .assign(
            lowest_price_per_mwh=data.gas_prc.iloc[0]['lowest_price_per_mwh'],
            highest_price_per_mwh=data.gas_prc.iloc[0]['highest_price_per_mwh'],
        )
        .merge(
            data.ele_prc,
            how='left', on=['hour']
        )
        .assign(
            **
            data.ele_prc
            ['euros_per_mwh']
            .agg(['mean', 'std', 'max'])
            .rename(lambda x: f'euros_per_mwh_{x}')
            .to_dict()
        )
        .merge(
            data.his_wea_tee_1
            .set_index(['county', 'hour'])
            .rename(lambda x: f'his_wea_{x}', axis=1),
            how='left', on=['county', 'hour']
        )
        .merge(
            data.his_wea_tee_1
            .filter(regex=r'^(?!enc_).*$')
            .drop(columns='hour')
            .groupby('county')
            .agg(['mean', 'std'])
            .pipe(flatten_column_index)
            .rename(lambda x: f'his_wea_{x}', axis=1),
            how='left', on=['county']
        )
        .merge(
            data.for_wea_tee_1
            .set_index(['county', 'datetime'])
            .rename(lambda x: f'for_wea_{x}', axis=1),
            how='left', on=['county', 'datetime']
        )
        .merge(
            data.for_wea_tee_1
            .filter(regex=r'^(?!sin_).*$')
            .filter(regex=r'^(?!cos_).*$')
            [data.for_wea_tee_1['datetime'].isin(data.targets['datetime'])]
            .drop(columns='datetime')
            .groupby('county')
            .agg(['mean', 'std'])
            .pipe(flatten_column_index)
            .rename(lambda x: f'for_wea_half_{x}', axis=1),
            how='left', on=['county']
        )
        .merge(
            data.for_wea_tee_1
            .filter(regex=r'^(?!sin_).*$')
            .filter(regex=r'^(?!cos_).*$')
            .drop(columns='datetime')
            .groupby('county')
            .agg(['mean', 'std'])
            .pipe(flatten_column_index)
            .rename(lambda x: f'for_wea_full_{x}', axis=1),
            how='left', on=['county']
        )
        .pipe(lambda df: df.assign(
            eic_per_cap=(df['eic_count'] / df['installed_capacity']),
            rev_tar_per_cap=(df['rev_target'] / df['installed_capacity']),
        ))
        .drop(columns=['datetime', 'row_id'])
        .astype('float32')
    )


### Form final data arrays for training Neural Network and Gradient-Boosted
# Decision Trees models
########################################################################
########################################################################

# Stack over all data blocks to create the final set of arrays for neural
# network training.
num_groups = len(data_list) - 2
X1 = stack([data.x1 for data in data_list[:-2]])
X2 = stack([data.x2 for data in data_list[:-2]])
X3 = stack([data.x3 for data in data_list[:-2]])
Y = (
    concatenate([data.y for data in data_list])[37: -11]
    .reshape(
        num_groups, 24, NUM_LAT, NUM_LON, len(HISTORY_COLUMNS)
    )
)
date = Series([data.targets['datetime'].min() for data in data_list[:-2]])
W = (
    cache['lat_lon']
    .set_index(['latitude', 'longitude'])['county']
    .map(cache['lat_lon'].groupby('county').size())
    .fillna(0.)
    .unstack()
    .values
    .astype(float32)
    .reshape(1, NUM_LAT, NUM_LON)
)
W = (
    (W / W.sum() * NUM_LAT * NUM_LON)
    .repeat(num_groups, axis=0)
)
data_nn = (X1, X2, X3, Y, W, date)

with open('data_nn.pkl', 'wb') as f:
    pickle.dump(data_nn, f)

print('Neural Network data arrays done and saved:')
print('\n'.join(map(stat, data_nn)))
# Print:
# NaN/Inf = 0, NonNum Cols = 0, shape = (634, 24, 5, 8, 14, 16)
# NaN/Inf = 0, NonNum Cols = 0, shape = (634, 6, 5, 8, 14, 16)
# NaN/Inf = 0, NonNum Cols = 0, shape = (634, 6, 8, 14, 7)
# NaN/Inf = 0, NonNum Cols = 0, shape = (634, 24, 8, 14, 7)
# NaN/Inf = 0, NonNum Cols = 0, shape = (634, 8, 14)
# NaN/Inf = 0, NonNum Cols = 1, shape = (634, 1)

# Concat over all data blocks to create the final dataframe for
# Gradient-Boosted Decision Trees training.
data_xgb = (
    concat([data.merged for data in data_list])
    .pipe(get_dummies, columns=['weekday', 'product_type', 'month', 'hour'])
    .astype('float32')
    .sort_index(axis=1)
    .reset_index(drop=True)
)

with open('data_xgb.pkl', 'wb') as f:
    pickle.dump(data_xgb, f)

print('GBDT dataframe done and saved:')
print(stat(data_xgb))
# Print:
# NaN/Inf = 7152, NonNum Cols = 0, shape = (2012496, 201)