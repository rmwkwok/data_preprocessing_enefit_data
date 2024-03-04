'''Utilities for creating new columns/features.'''

from numpy import pi, sin, cos
from pandas import DataFrame, to_datetime


# for generating holiday feature
HOLIDAYS = to_datetime([
    '2021-Jan-1', '2021-Feb-24', '2021-Apr-2', '2021-Apr-4', '2021-May-1', '2021-May-23', '2021-Jun-23', '2021-Jun-24', '2021-Aug-20', '2021-Dec-24', '2021-Dec-25', '2021-Dec-26',
    '2022-Jan-1', '2022-Feb-24', '2022-Apr-15', '2022-Apr-17', '2022-May-1', '2022-Jun-5', '2022-Jun-23', '2022-Jun-24', '2022-Aug-20', '2022-Dec-24', '2022-Dec-25', '2022-Dec-26',
    '2023-Jan-1', '2023-Feb-24', '2023-Apr-7', '2023-Apr-9', '2023-May-1', '2023-May-28', '2023-Jun-23', '2023-Jun-24', '2023-Aug-20', '2023-Dec-24', '2023-Dec-25', '2023-Dec-26',
    '2024-Jan-1', '2024-Feb-24', '2024-Mar-29', '2024-Mar-31', '2024-May-1', '2024-May-19', '2024-Jun-23', '2024-Jun-24', '2024-Aug-20', '2024-Dec-24', '2024-Dec-25', '2024-Dec-26',
]).date

# for conversion of "day" into "day of lunar calendar"
# 2023-02-13 18:00 is the time of the Moon Phase's third quater
# https://www.timeanddate.com/moon/phases/latvia/riga?year=2023
LUNAR_ZERO = to_datetime('2023-02-13 18:00:00')
LUNAR_PERIOD = 29.53059  # in days


def datetime_split(
    df: DataFrame, datetime_column: str,
    parts: tuple[str, ...]=('hour', ), encode: bool=False,
) -> DataFrame:
    '''Split datetime into parts.

    Args:
        df: pandas.DataFrame,
        datetime_column: str
        encode: if `False`, return the parts as is. If `True`,
            encode each part as its sine and cosine components.
        parts: tuple[str, ...]=('hour', )
            Any combination of ['year', 'month', 'weekday', 'day',
            'lunarday', 'hour']
    '''
    if isinstance(parts, str):
        parts = (parts, )

    dt = df[datetime_column]

    lunarday = (dt - LUNAR_ZERO)
    lunarday = (lunarday.dt.days + lunarday.dt.seconds / 86400) % LUNAR_PERIOD

    parts = {k: v for k, v in {
        'day': (dt.dt.day, 1, dt.dt.daysinmonth),
        'hour': (dt.dt.hour, 0., 24.),
        'year': (dt.dt.year, 2021., 100.),
        'month': (dt.dt.month, 1., 12.),
        'weekday': (dt.dt.weekday, 0., 7.),
        'lunarday': (lunarday, 0, LUNAR_PERIOD),
    }.items() if k in parts}

    cols = {}
    for part, (data, offset, period) in parts.items():
        if encode:
            cols[f'cos_{part}'] = cos(2 * pi * (data - offset) / period)
            cols[f'sin_{part}'] = sin(2 * pi * (data - offset) / period)
        else:
            cols[f'{part}'] = data

    return df.assign(**cols)


def convert_latlon(df: DataFrame) -> DataFrame:
    '''Avoid precision problem.'''
    return df.assign(
        latitude=df['latitude'].multiply(10).round(0).astype('int'),
        longitude=df['longitude'].multiply(10).round(0).astype('int'),
    )


def unify_weather_data(df: DataFrame, is_history: bool=True) -> DataFrame:
    '''Unify weather data representations.

    Args:
        df: pandas.DataFrame
        is_history: bool=True
            If `True`, expect history weather data, otherwise,
            forecast weather data.
    '''
    if is_history:
        # Wind
        wind_r = df['windspeed_10m']
        wind_t = df['winddirection_10m'] * pi / 180
        sin_wind_t = sin(wind_t)
        cos_wind_t = cos(wind_t)
        wind_u = wind_r * cos_wind_t
        wind_v = wind_r * sin_wind_t

        rain = df['rain'] / 1000  # convert from mm to m
        snow = df['snowfall'] / 1000  # from mm to m
        total = rain + snow
    else:
        wind_u = df['10_metre_u_wind_component']
        wind_v = df['10_metre_v_wind_component']
        wind_r = (wind_u ** 2 + wind_v ** 2) ** .5
        cos_wind_t = wind_u / (wind_r + 1e-7)
        sin_wind_t = wind_v / (wind_r + 1e-7)

        snow = df['snowfall']
        total = df['total_precipitation']
        rain = (total - snow).clip(0.)
    return (
        df
        .drop(columns='winddirection_10m', errors='ignore')
        .assign(**{
            '10_metre_u_wind_component': wind_u,
            '10_metre_v_wind_component': wind_v,
            'windspeed_10m': wind_r,
            'cos_winddirection_10m': cos_wind_t,
            'sin_winddirection_10m': sin_wind_t,

            'rain': rain,
            'snowfall': snow,
            'total_precipitation': total,
        })
    )


def convert_datetime(df: DataFrame, datetime_columns: list[str]) -> DataFrame:
    '''Convert datetime from text format to datetime format.

    Args:
        df: pandas.DataFrame
        datetime_columns: list[str]
            A list of datetime columns to convert.
    '''
    for column in datetime_columns:
        if column in df:
            df[column] = to_datetime(df[column])
    return df


def add_holiday(df: DataFrame, datetime_column: str) -> DataFrame:
    '''Add a column feature for whether it is a holiday.

    Args:
        df: pandas.DataFrame
        datetime_columns: str
            The datetime column to be considered for.
    '''
    return df.assign(
        is_holiday=(
            df[datetime_column].dt.date.isin(HOLIDAYS) |
            df[datetime_column].dt.weekday.isin([5, 6])
        )
    )

