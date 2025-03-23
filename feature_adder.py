import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler


 
 
def add_time_since_t0_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the time since t0 feature to the data.

    Args:
        df (DataFrame): The dataframe containing the data.

    Returns:
        DataFrame: A dataframe containing the time since t0 feature.
    """
    df_copy = df.copy()
    t0 = df_copy.index[0]
    df_copy['Time_Since_t0'] = (df_copy.index - t0).total_seconds() 
    #df_copy['Time_Since_t0_n'] = (df_copy['Time_Since_t0'] - df_copy['Time_Since_t0'].mean()) / df_copy['Time_Since_t0'].std()
    
    y_s = df_copy['Time_Since_t0']
    y_ts  = TimeSeries.from_series(y_s)
    scaler1 = Scaler()
    y_n = scaler1.fit_transform(y_ts)
    df_copy['Time_Since_t0_n'] = y_n.values()
   
    return df_copy
  



def add_num_not_Nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the number of non-NaN values feature to the data.

    Args:
        df (DataFrame): The dataframe containing the data.

    Returns:
        DataFrame: A dataframe containing the number of non-NaN values feature.
    """
    df_copy = df.copy()
    df_copy['Num_Not_Nan'] = df_copy.notna().sum(axis=1) - 2
    
    #y_s = df_copy['Num_Not_Nan']
    #y_ts  = TimeSeries.from_series(y_s)
    #scaler1 = Scaler()
    #y_n = scaler1.fit_transform(y_ts)
    #df_copy['Num_Not_Nan_n'] = y_n.values()
    
    
    return df_copy

def add_num_Nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the number of NaN values feature to the data.

    Args:
        df (DataFrame): The dataframe containing the data.

    Returns:
        DataFrame: A dataframe containing the number of NaN values feature.
    """
    df_copy = df.copy()
    df_copy['Num_Nan'] = df_copy.isna().sum(axis=1)
    
    #y_s = df_copy['Num_Nan']
    #y_ts  = TimeSeries.from_series(y_s)
    #scaler1 = Scaler()
    #y_n = scaler1.fit_transform(y_ts)
    #df_copy['Num_Nan_n'] = y_n.values()
    
    
    return df_copy


def add_num_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the total number of bytes for each row's data (excluding the first two columns)
    and store it in a new column.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: A dataframe with an added 'Num_Bytes' column containing total byte size.
    """
    df_copy = df.copy()

    # Select all columns except the first two
    selected_columns = df_copy.iloc[:, 2:]

    # Compute total bytes for each row
    df_copy['Num_Bytes'] = selected_columns.astype(str).apply(lambda row: sum(row.str.encode('utf-8').map(len)), axis=1)
    
       
    #y_s = df_copy['Num_Bytes']
    #y_ts  = TimeSeries.from_series(y_s)
    #scaler1 = Scaler()
    #y_n = scaler1.fit_transform(y_ts)
    #df_copy['Num_Bytes_n'] = y_n.values()
    
    
    return df_copy



import pandas as pd
from darts.utils.timeseries_generation import datetime_attribute_timeseries

def add_year_month_week(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure index is datetime and remove timezone information
    datetime_index = df.index.tz_localize(None) if df.index.tz else df.index

    # Generate time-based features using Darts
    year_series = datetime_attribute_timeseries(time_index=datetime_index, attribute="year", one_hot=False)
    month_series = datetime_attribute_timeseries(time_index=datetime_index, attribute="month", one_hot=True)
    weekday_series = datetime_attribute_timeseries(time_index=datetime_index, attribute="weekday", one_hot=True)

    # Convert TimeSeries to DataFrame (Fixing Deprecation Warning)
    year_df = year_series.to_dataframe()  # Updated method
    month_df = month_series.to_dataframe()
    weekday_df = weekday_series.to_dataframe()

    # Remove timezone information if any
    for df_part in [year_df, month_df, weekday_df]:
        df_part.index = df_part.index.tz_localize(None) if df_part.index.tz else df_part.index

    # Concatenate original dataframe with generated features
    df = pd.concat([df.tz_localize(None) if df.index.tz else df, year_df, month_df, weekday_df], axis=1)

    return df

'''def add_year_month_week(df)->pd.DataFrame:
    # Convert to timezone-naive DatetimeIndex
    datetime_index = df.index.tz_localize(None)  # Remove UTC timezone

    # Generate time-based features
    year_series = datetime_attribute_timeseries(time_index=datetime_index, attribute="year", one_hot=False)

    month_series = datetime_attribute_timeseries(time_index=datetime_index, attribute="month", one_hot=True)

    weekday_series = datetime_attribute_timeseries(time_index=datetime_index, attribute="weekday", one_hot=True)

    # Convert TimeSeries to DataFrame
    year_df = year_series.pd_dataframe()
    month_df = month_series.pd_dataframe()
    weekday_df = weekday_series.pd_dataframe()

    # Concatenate the dataframes
    # Ensure all dataframes have the same timezone-naive index
    year_df.index = year_df.index.tz_localize(None)
    month_df.index = month_df.index.tz_localize(None)
    weekday_df.index = weekday_df.index.tz_localize(None)

    # Concatenate the dataframes
    df = pd.concat([df.tz_localize(None), year_df, month_df, weekday_df], axis=1)
    return df'''


def add_austevoll_frost_temperature_feature(df: pd.DataFrame) -> pd.DataFrame:
    frost_temperature_df = pd.read_csv("row_data\\frost_temperature_data.csv")
    frost_temperature_df["Time"] = pd.to_datetime(frost_temperature_df["Time"])
    frost_temperature_df.set_index("Time", inplace=True)
    
    df_copy = df.copy()
    
    frost_temperature_df = frost_temperature_df.resample('1h').mean().ffill()
    
    frost_temperature_df.index = frost_temperature_df.index.tz_localize(None)

    df_copy['Temperature'] = frost_temperature_df.loc[df_copy.index[0]:df_copy.index[-1]].Temperature
    
    
    #y_s = df_copy['Temperature']
    #y_ts  = TimeSeries.from_series(y_s)
    #scaler1 = Scaler()
    #y_n = scaler1.fit_transform(y_ts)
    #df_copy['Temperature_n'] = y_n.values()
    
    return df_copy


def temp_sum_since_t0(df: pd.DataFrame)->pd.DataFrame:
    copy_df = df.copy()
    copy_df['Temperature_sum'] = copy_df['Temperature'].cumsum()
    return copy_df



# 2. Gaussian Smoothing
from scipy.ndimage import gaussian_filter1d

def smooth_data_4(df, column, sigma) -> pd.DataFrame:
    df_copy = df.copy()
    
    y = df_copy[column].to_numpy()
    df_copy['smooth_'+column] = gaussian_filter1d(y, sigma=sigma)
    yhat_df = df_copy[['smooth_'+column]]
    V_O_df = df_copy[[column]]

    plt.figure(figsize=(10, 6))
    plt.plot(V_O_df, '.', label='Original Voltage')
    plt.plot(yhat_df, '.', label='Smoothed Voltage (Gaussian)')
    
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return df_copy
    






def add_fourier_feature(df: pd.DataFrame) -> pd.DataFrame:
    # Load Fourier feature dataset
    f_t_df = pd.read_csv('row_data\\fourier_features_2024.csv')
    f_t_df.set_index('Time', inplace=True)
    f_t_df.index = pd.to_datetime(f_t_df.index)

    # Copy input DataFrame
    copy_df = df.copy()

    # Ensure f_t_df has the necessary columns
    cols = ['sin_month', 'cos_month', 'sin_week', 
                     'cos_week', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour']
    
    # Create time-based columns for merging
    for df_ in [f_t_df, copy_df]:
        df_['month'] = df_.index.month
        df_['week'] = df_.index.isocalendar().week
        df_['day'] = df_.index.dayofyear
        df_['hour'] = df_.index.hour


    copy_df.reset_index(inplace=True)
    
    # Merge based on time-based columns
    copy_df = copy_df.merge(
        f_t_df[['month', 'week', 'day', 'hour'] + cols], 
        on=['month', 'week', 'day', 'hour'], 
        how='left'
    )
    
    # Drop intermediate time-based columns
    copy_df.drop(columns=['month', 'week', 'day', 'hour'], inplace=True)
    
    copy_df.set_index('Time', inplace=True)

    return copy_df





def normalize_data(df: pd.DataFrame, column : str) -> pd.DataFrame:
    df_copy = df.copy()
    
    scaler1 = Scaler()
    
    v_i = df_copy[column]
    y_n  = TimeSeries.from_series(v_i)
    y_n = scaler1.fit_transform(y_n)
    
    df_copy[column+"_n"] = y_n.values()
    
    return df_copy, scaler1


from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
import pandas as pd

def normalize_data(df: pd.DataFrame, column: str):
    df_copy = df.copy()
    
    scaler1 = Scaler()
    
    v_i = df_copy[column]
    y_n  = TimeSeries.from_series(v_i)
    y_n = scaler1.fit_transform(y_n)
    
    df_copy[column + "_n"] = y_n.values()
    
    return df_copy, scaler1


from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
import pandas as pd


def normalize_data(df: pd.DataFrame, column: str):
    df_copy = df.copy()
    
    scaler1 = Scaler()
    
    v_i = df_copy[column]
    y_n  = TimeSeries.from_series(v_i)
    y_n = scaler1.fit_transform(y_n)
    
    df_copy[column + "_n"] = y_n.values()
    
    return df_copy, scaler1

def inverse_transform(df: pd.DataFrame, column: str, scaler: Scaler) -> pd.DataFrame:
    df_copy = df.copy()
    
    # Convert normalized values back to TimeSeries
    y_n = TimeSeries.from_series(df_copy[column])
    
    # Inverse transform
    y_original = scaler.inverse_transform(y_n)
    
    # Assign back to DataFrame
    df_copy[column.replace("_n", "") + "_original"] = y_original.values()
    
    return df_copy




