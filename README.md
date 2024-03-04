# Data Preprocessing for Enefit's Kaggle dataset

This demonstrates the use of method chaining in processing multiple dataframes, and merge the processed dataframes into a final dataframe. The preprocessing functions are intentionally saved in multiple utilities scripts for clearity, and the scripts' names explain their purposes.

All processed dataframes are then combined into a dataset for training Neural Network models and a dataset for training Gradient Boosted Decision Trees models.

To run this script, the dataset needs to be downloaded from [Kaggle](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers) and unzipped to the directory of the script. On success, two pickle files will be produced and the following printed on screen:

```
$python3 main.py
Data Readout Done.
Preprocessing begins.
100%|██████████████████████| 636/636 [01:13<00:00,  8.60it/s]
Neural Network data arrays done and saved:
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 24, 5, 8, 14, 16)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 6, 5, 8, 14, 16)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 6, 8, 14, 7)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 24, 8, 14, 7)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 8, 14)
NaN/Inf = 0, NonNum Cols = 1, shape = (634, 1)
GBDT dataframe done and saved:
NaN/Inf = 7152, NonNum Cols = 0, shape = (2012496, 201)
```

For those interested, `(634, 24, 5, 8, 14, 16)` is an array of 634 days, 24 time-series per day, 5 steps per series, 8 * 14 grids per step, and 16 features per grid. This and other arrays were used to train a U-net-like, conditional, 2D+3D Convolutional network, trying to find relation between forecasted and recorded weather data.

A `Data` class is defined in accordance with the competition's way of providing the data - each object expects a day-worth of data in several dataframes. While the current script has one chain of methods for each of the dataframes, it is also possible to have one chain for all of them, by having the `Data` class inheriting the `FocusableBase` defined but unused in `data_class.py`. `FocusableBase` lets us designate a dataframe-of-focus and redirect all method calls to it. `FocusableBase` was made for fun :) only.

## Method chaining examples:

```python
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
```

```python
data.merged = (
    data.targets
    .merge(
        data.rev_tar,
        how='left', on=['county', 'is_business', 'product_type', 'is_consumption', 'hour']
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
    ...
```