import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from dateutil.relativedelta import relativedelta
from math import factorial
from scipy.ndimage import gaussian_filter1d




def time_period(df, start=None, end=None):
    """
    Calculate the difference in months, days, hours, seconds, and milliseconds
    between the start and start_date.

    Args:
      df (DataFrame): The dataframe containing the data.
      start (datetime, optional): The start datetime. Defaults to None.
      end (datetime, optional): The end datetime. Defaults to None.

    Returns:
      tuple: A tuple containing months, days, hours, seconds, and milliseconds.
    """
    
    if start is None:
        start = df.index[0]
    
    if end is None:
        end = df.index[-1]

    # Calculate the difference using relativedelta
    difference = relativedelta(end, start)

    months = difference.months
    days = difference.days
    hours = difference.hours
    seconds = difference.seconds
    milliseconds = difference.microseconds // 1000
    
    return months, days, hours, seconds, milliseconds
  
  
def plot_dfs(dfs: list, format=".", start=None, end=None, title=None):
  """
  Visualizes time series data for all columns in the data frames

  Args:
    dfs (tuple of pandas.DataFrame) - contains the data frames to plot
    format - line style when plotting the graph
    start - first time step to plot
    end - last time step to plot
  """

  # Setup dimensions of the graph figure
  plt.figure(figsize=(10, 6))

  for data_frame in dfs:
      
    for column in data_frame.columns:
      # Plot the time series data for each column
      plt.plot(data_frame[column][start:end], format, label=column)
      
      plt.title(title)
     
      # Label the x-axis
      plt.xlabel("Time")
      
      # Label the y-axis
      plt.ylabel(str(column))

  # Overlay a grid on the graph
  plt.grid(True)

  # Add a legend
  plt.legend()

  # Draw the graph on screen
  plt.show()











'''def get_hourly_temperature(df, long : float = 5.2639, lat : float = 60.090533, minutes=None, hours=None, days=None, weeks=None, months=None, years=None ):
  client = client_met.METClient()
  
  location = Location(longitude=long, latitude=lat)
  
  from_start = datetime.datetime.strptime(str(df.index[0]), '%Y-%m-%d %H:%M:%S%z')
  to_end = datetime.datetime.strptime(str(df.index[-1]), '%Y-%m-%d %H:%M:%S%z') + datetime.timedelta(days=1, hours=1, seconds=1)

  
  print(from_start)
  print(to_end)
  
  obs = client.fetch_observations(location, start=from_start, end= to_end)
  
  frost_temp_df = pd.DataFrame([
    {
        "Time": datetime.datetime.strptime(str(point.timestamp), '%Y-%m-%d %H:%M:%S%z'),
        "temperature": point.temperature
    }
    for point in obs.data
  ])
  frost_temp_df.set_index('Time', inplace=True)
  
  if minutes is not None:
    frost_temp_df = frost_temp_df['temperature'].resample(f'{str(minutes)}min').mean()
  elif hours is not None:
    frost_temp_df = frost_temp_df['temperature'].resample(f'{str(hours)}h').mean()
  elif days is not None:
    frost_temp_df = frost_temp_df['temperature'].resample(f'{str(days)}d').mean()
  elif weeks is not None:
    frost_temp_df = frost_temp_df['temperature'].resample(f'{str(weeks)}W').mean()
  elif months is not None:
    frost_temp_df = frost_temp_df['temperature'].resample(f'{str(months)}ME').mean()
  elif years is not None:
    frost_temp_df = frost_temp_df['temperature'].resample(f'{str(years)}YE').mean()
  else:
    raise ValueError("At least one time period must be specified.")
  
  return frost_temp_df.to_frame()'''



def plot_all_df_colones_in_different_plots(df):
  """
  Visualizes time series data for all columns in the data frames

  Args:
    df (DataFrame) - contains the data frames to plot
  """
  
  # Setup dimensions of the graph figure
  plt.figure(figsize=(10, 6))
  
  for column in df.columns:
    # Plot the time series data for each column
    plt.plot(df[column], label=column, marker='.')
    
    # Label the x-axis
    plt.xlabel("Time")
    
    # Label the y-axis
    plt.ylabel(str(column))
    
    # Overlay a grid on the graph
    plt.grid(True)
    
    # Add a legend
    plt.legend()
    
    # Draw the graph on screen
    plt.show()




def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
 
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
  
  
  
  


def plot_smoothed_data_1(df, window_size, order):

  df_copy = df.copy()
  y = df_copy['System_Parameters.Input_Voltage'] 
  y = y.to_numpy()
  
  yhat = savitzky_golay(y, window_size,order) # window size 51, polynomial order 3
  
  df_copy['yhat'] = yhat
  yhat_df = df_copy[['yhat']]
  V_O_df = df_copy[['System_Parameters.Input_Voltage']]
  
  plt.figure(figsize=(10, 6))
  
  plt.plot(V_O_df, '.', label='Original Voltage')
  plt.plot(yhat_df, '.', label='Smoothed Voltage')
  
  plt.xlabel("Time")
  plt.ylabel("Voltage")
  plt.grid(True)
  plt.legend()
  plt.show()
  
def plot_smoothed_data_2(df, window_size, min_periods):

  df_copy = df.copy()
  y = df_copy['System_Parameters.Input_Voltage'] 
  y = y.to_numpy()
  df_copy['yhat'] = df_copy['System_Parameters.Input_Voltage'].rolling(window=window_size, min_periods=min_periods).max()

  
  yhat_df = df_copy[['yhat']]
  V_O_df = df_copy[['System_Parameters.Input_Voltage']]
  
  plt.figure(figsize=(10, 6))
  
  plt.plot(V_O_df, '.', label='Original Voltage')
  plt.plot(yhat_df, '.', label='Smoothed Voltage')
  
  plt.xlabel("Time")
  plt.ylabel("Voltage")
  plt.grid(True)
  plt.legend()
  plt.show()
  
  
  
def plot_smoothed_data_3(df, span):
    df_copy = df.copy()
    df_copy['yhat'] = df_copy['System_Parameters.Input_Voltage'].ewm(span=span).mean()
    
    yhat_df = df_copy[['yhat']]
    V_O_df = df_copy[['System_Parameters.Input_Voltage']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(V_O_df, '.', label='Original Voltage')
    plt.plot(yhat_df, '.', label='Smoothed Voltage (EMA)')
    
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_smoothed_data_4(df, sigma) -> pd.DataFrame:
    df_copy = df.copy()
    y = df_copy['System_Parameters.Input_Voltage'].to_numpy()
    
    df_copy['smooth_input_voltage'] = gaussian_filter1d(y, sigma=sigma)
    
    yhat_df = df_copy[['smooth_input_voltage']]
    V_O_df = df_copy[['System_Parameters.Input_Voltage']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(V_O_df, '.', label='Original Voltage')
    plt.plot(yhat_df, '.', label='Smoothed Voltage (Gaussian)')
    
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return df_copy