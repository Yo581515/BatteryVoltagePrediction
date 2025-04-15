from feature_adder import add_time_since_t0_feature
from feature_adder import add_year_month_week
from feature_adder import add_fourier_feature
from feature_adder import add_num_not_Nan
from feature_adder import add_num_Nan
from feature_adder import add_num_bytes
from feature_adder import add_austevoll_frost_temperature_feature
from feature_adder import temp_sum_since_t0
from feature_adder import add_voltage_features
from feature_adder import smooth_df_voltage_fun_df

import matplotlib.pyplot as plt  
plt.style.use('./deeplearning.mplstyle')
import pandas as pd
import datetime

list_features = [
        'Num_Not_Nan',
        #'Num_Not_Nan_n',
        'Num_Nan',
        #'Num_Nan_n',
        'Num_Bytes',
        #'Num_Bytes_n',
        
        
        'Time_Since_t0', 
        'Time_Since_t0_n',
        'Hours_Since_t0',
        'Days_Since_t0', 
        'Weeks_Since_t0',
        'Months_Since_t0',
        
        
        'year', 
        'month_0', 
        'month_1',
        'month_2', 
        'month_3', 
        'month_4', 
        'month_5', 
        'month_6', 
        'month_7',
        'month_8', 
        'month_9', 
        'month_10', 
        'month_11', 
        'weekday_0', 
        'weekday_1',
        'weekday_2', 
        'weekday_3', 
        'weekday_4', 
        'weekday_5', 
        'weekday_6',
        'Temperature',
        #'Temperature_n',
        'Temperature_sum', 
        
        
        'sin_month', 
        'cos_month', 
        'sin_week', 
        'cos_week',
        'sin_day', 
        'cos_day', 
        'sin_hour', 
        'cos_hour',
        
        'System_Parameters.Input_Voltage',
        'Battery_Level_Fitted',
        'Battery_Level_Derivative',
    
        'Voltage_Lag1',
        'Voltage_Lead1',
        'Voltage_Diff',
        'Voltage_Change_Rate',
        'Rolling_Mean_Voltage',
        'Rolling_Max_Voltage',
        'Rolling_Std_Voltage',
    ]


'''def setup_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = add_time_since_t0_feature(df)
    
    df = add_year_month_week(df)
    
    df = add_austevoll_temperature_feature(df)
    
    return df'''



def setup_as_data(df: pd.DataFrame, smooth_sigma: int, list_segments: list[int], r_h=1) -> pd.DataFrame:
    Austevoll_Sor_df = df.copy()
        
    Austevoll_Sor_df = add_num_not_Nan(Austevoll_Sor_df)
    Austevoll_Sor_df = add_num_Nan(Austevoll_Sor_df)
    Austevoll_Sor_df = add_num_bytes(Austevoll_Sor_df)

    as_t_1 = datetime.datetime.strptime('2022-10-24 15:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_2 = datetime.datetime.strptime('2022-12-15 15:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_3 = datetime.datetime.strptime('2023-02-22 12:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_4 = datetime.datetime.strptime('2023-06-08 03:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_5 = datetime.datetime.strptime('2023-08-29 04:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_6 = datetime.datetime.strptime('2023-11-13 11:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_7 = datetime.datetime.strptime('2024-02-14 12:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_8 = datetime.datetime.strptime('2024-05-15 11:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_9 = datetime.datetime.strptime('2024-08-01 00:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_10 = datetime.datetime.strptime('2024-11-12 16:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')

    as_t_1_e = datetime.datetime.strptime('2022-10-24 15:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_2_e = datetime.datetime.strptime('2022-12-15 15:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_3_e= datetime.datetime.strptime('2023-03-01 00:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_4_e = datetime.datetime.strptime('2023-06-08 08:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_5_e = datetime.datetime.strptime('2023-08-29 08:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_6_e = datetime.datetime.strptime('2023-11-13 16:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_7_e = datetime.datetime.strptime('2024-02-14 12:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_8_e = datetime.datetime.strptime('2024-05-15 11:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_9_e = datetime.datetime.strptime('2024-08-01 00:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_t_10_e = datetime.datetime.strptime('2024-11-12 16:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')


    as_seg_1_e = Austevoll_Sor_df.loc[as_t_1_e:as_t_2 - datetime.timedelta(seconds=1)]
    
    as_seg_2_end_time = datetime.datetime.strptime('2023-02-16 15:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    as_seg_2_e = Austevoll_Sor_df.loc[as_t_2_e:as_seg_2_end_time - datetime.timedelta(seconds=1)]
    
    
    as_seg_3_e = Austevoll_Sor_df.loc[as_t_3_e:as_t_4 - datetime.timedelta(seconds=1)]
    as_seg_3_e = as_seg_3_e.loc[as_seg_3_e['System_Parameters.Input_Voltage'] <= 12.4]

    as_seg_4_e = Austevoll_Sor_df.loc[as_t_4_e:as_t_5 - datetime.timedelta(seconds=1)]
    as_seg_4_e = as_seg_4_e.loc[as_seg_4_e['System_Parameters.Input_Voltage'] <= 12.5]

    as_seg_5_e = Austevoll_Sor_df.loc[as_t_5_e:as_t_6 - datetime.timedelta(seconds=1)]
    as_seg_6_e = Austevoll_Sor_df.loc[as_t_6_e:as_t_7 - datetime.timedelta(seconds=1)]

    as_seg_7_e = Austevoll_Sor_df.loc[as_t_7_e:as_t_8 - datetime.timedelta(seconds=1)]
    as_seg_7_e = as_seg_7_e.loc[as_seg_7_e['System_Parameters.Input_Voltage'] <= 12.4]

    as_seg_8_e = Austevoll_Sor_df.loc[as_t_8_e:as_t_9 - datetime.timedelta(seconds=1)]
   # Uncomment if needed:
    # as_seg_8_e = as_seg_8_e.loc[as_seg_8_e['System_Parameters.Input_Voltage'] <= 12.2]

    as_seg_9_e = Austevoll_Sor_df.loc[as_t_9_e:as_t_10 - datetime.timedelta(seconds=1)]
    as_seg_10_e = Austevoll_Sor_df.loc[as_t_10_e:]

    
    hours = r_h
    if 1 in list_segments:
        as_seg_1_e = as_seg_1_e.resample(f'{hours}h').max().ffill()
    if 2 in list_segments:
        as_seg_2_e = as_seg_2_e.resample(f'{hours}h').max().ffill()
    if 3 in list_segments:
        as_seg_3_e = as_seg_3_e.resample(f'{hours}h').max().ffill()
    if 4 in list_segments:
        as_seg_4_e = as_seg_4_e.resample(f'{hours}h').max().ffill()
    if 5 in list_segments:
        as_seg_5_e = as_seg_5_e.resample(f'{hours}h').mean().ffill()
    if 6 in list_segments:
        as_seg_6_e = as_seg_6_e.resample(f'{hours}h').max().ffill()
    if 7 in list_segments:
        as_seg_7_e = as_seg_7_e.resample(f'{hours}h').max().ffill()
    if 8 in list_segments:
        as_seg_8_e = as_seg_8_e.resample(f'{hours}h').max().ffill()
    if 9 in list_segments:
        as_seg_9_e = as_seg_9_e.resample(f'{hours}h').max().ffill()
    if 10 in list_segments:
        as_seg_10_e = as_seg_10_e.resample(f'{hours}h').max().ffill()
        
    
    if 1 in list_segments:
        as_seg_1_e = add_time_since_t0_feature(as_seg_1_e)
        as_seg_1_e = add_year_month_week(as_seg_1_e)
        as_seg_1_e = add_austevoll_frost_temperature_feature(as_seg_1_e)
        as_seg_1_e = add_fourier_feature(as_seg_1_e)
        as_seg_1_e = temp_sum_since_t0(as_seg_1_e)
        
        as_seg_1_e = smooth_df_voltage_fun_df(as_seg_1_e)
        
        as_seg_1_e = add_voltage_features(as_seg_1_e)
        as_seg_1_e = as_seg_1_e[list_features]
    
    if 2 in list_segments:
        as_seg_2_e = add_time_since_t0_feature(as_seg_2_e)
        as_seg_2_e = add_year_month_week(as_seg_2_e)
        as_seg_2_e = add_austevoll_frost_temperature_feature(as_seg_2_e)
        as_seg_2_e = temp_sum_since_t0(as_seg_2_e)
        as_seg_2_e = add_fourier_feature(as_seg_2_e)
        
        as_seg_2_e = smooth_df_voltage_fun_df(as_seg_2_e)
        
        as_seg_2_e = add_voltage_features(as_seg_2_e)
        as_seg_2_e = as_seg_2_e[list_features]
    
    if 3 in list_segments:
        as_seg_3_e = add_time_since_t0_feature(as_seg_3_e)
        as_seg_3_e = add_year_month_week(as_seg_3_e)
        as_seg_3_e = add_austevoll_frost_temperature_feature(as_seg_3_e)
        as_seg_3_e = temp_sum_since_t0(as_seg_3_e)
        as_seg_3_e = add_fourier_feature(as_seg_3_e)
        
        as_seg_3_e = smooth_df_voltage_fun_df(as_seg_3_e)
        
        as_seg_3_e = add_voltage_features(as_seg_3_e, window=12)
        as_seg_3_e = as_seg_3_e[list_features]
        
    if 4 in list_segments:
        as_seg_4_e = add_time_since_t0_feature(as_seg_4_e)
        as_seg_4_e = add_year_month_week(as_seg_4_e)
        as_seg_4_e = add_austevoll_frost_temperature_feature(as_seg_4_e)
        as_seg_4_e = temp_sum_since_t0(as_seg_4_e)
        as_seg_4_e = add_fourier_feature(as_seg_4_e)
        
        as_seg_4_e = smooth_df_voltage_fun_df(as_seg_4_e)
        
        as_seg_4_e = add_voltage_features(as_seg_4_e)
        as_seg_4_e = as_seg_4_e[list_features]
        
    if 5 in list_segments:
        as_seg_5_e = add_time_since_t0_feature(as_seg_5_e)
        as_seg_5_e = add_year_month_week(as_seg_5_e)
        as_seg_5_e = add_austevoll_frost_temperature_feature(as_seg_5_e)
        as_seg_5_e = temp_sum_since_t0(as_seg_5_e)
        as_seg_5_e = add_fourier_feature(as_seg_5_e)
        
        as_seg_5_e = smooth_df_voltage_fun_df(as_seg_5_e)
        
        as_seg_5_e = add_voltage_features(as_seg_5_e)
        as_seg_5_e = as_seg_5_e[list_features]
        
    if 6 in list_segments:
        as_seg_6_e = add_time_since_t0_feature(as_seg_6_e)
        as_seg_6_e = add_year_month_week(as_seg_6_e)
        as_seg_6_e = add_austevoll_frost_temperature_feature(as_seg_6_e)
        as_seg_6_e = temp_sum_since_t0(as_seg_6_e)
        as_seg_6_e = add_fourier_feature(as_seg_6_e)
        
        as_seg_6_e = smooth_df_voltage_fun_df(as_seg_6_e)
        
        as_seg_6_e = add_voltage_features(as_seg_6_e)
        as_seg_6_e = as_seg_6_e[list_features]
        
    if 7 in list_segments:
        as_seg_7_e = add_time_since_t0_feature(as_seg_7_e)
        as_seg_7_e = add_year_month_week(as_seg_7_e)
        as_seg_7_e = add_austevoll_frost_temperature_feature(as_seg_7_e)
        as_seg_7_e = temp_sum_since_t0(as_seg_7_e)
        as_seg_7_e = add_fourier_feature(as_seg_7_e)
        
        as_seg_7_e = smooth_df_voltage_fun_df(as_seg_7_e)
        
        as_seg_7_e = add_voltage_features(as_seg_7_e)
        as_seg_7_e = as_seg_7_e[list_features]
        
    if 8 in list_segments:
        as_seg_8_e = add_time_since_t0_feature(as_seg_8_e)
        as_seg_8_e = add_year_month_week(as_seg_8_e)
        as_seg_8_e = add_austevoll_frost_temperature_feature(as_seg_8_e)
        as_seg_8_e = temp_sum_since_t0(as_seg_8_e)
        as_seg_8_e = add_fourier_feature(as_seg_8_e)
        
        as_seg_8_e = smooth_df_voltage_fun_df(as_seg_8_e)
        
        as_seg_8_e = add_voltage_features(as_seg_8_e)
        as_seg_8_e = as_seg_8_e[list_features]
        
    if 9 in list_segments:
        as_seg_9_e = add_time_since_t0_feature(as_seg_9_e)
        as_seg_9_e = add_year_month_week(as_seg_9_e)
        as_seg_9_e = add_austevoll_frost_temperature_feature(as_seg_9_e)
        as_seg_9_e = temp_sum_since_t0(as_seg_9_e)
        as_seg_9_e = add_fourier_feature(as_seg_9_e)
        
        as_seg_9_e = smooth_df_voltage_fun_df(as_seg_9_e)
        
        as_seg_9_e = add_voltage_features(as_seg_9_e)
        as_seg_9_e = as_seg_9_e[list_features]
        
    if 10 in list_segments:
        as_seg_10_e = add_time_since_t0_feature(as_seg_10_e)
        as_seg_10_e = add_year_month_week(as_seg_10_e)
        as_seg_10_e = add_austevoll_frost_temperature_feature(as_seg_10_e)
        as_seg_10_e = temp_sum_since_t0(as_seg_10_e)
        as_seg_10_e = add_fourier_feature(as_seg_10_e)
        
        as_seg_10_e = smooth_df_voltage_fun_df(as_seg_10_e)
        
        as_seg_10_e = add_voltage_features(as_seg_10_e)
        as_seg_10_e = as_seg_10_e[list_features]
    
    segments = {
        1: as_seg_1_e,
        2: as_seg_2_e,
        3: as_seg_3_e,
        4: as_seg_4_e,
        5: as_seg_5_e,
        6: as_seg_6_e,
        7: as_seg_7_e,
        8: as_seg_8_e,
        9: as_seg_9_e,
        10: as_seg_10_e
    }
    return {key: segments[key] for key in list_segments if key in segments}



def setup_an_data(df: pd.DataFrame, smooth_sigma: int, list_segments: list[int], r_h=1) -> pd.DataFrame:
    #an_seg_1_end_time = datetime.datetime.strptime('2023-08-12 00:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    Austevoll_Nord_df = df.copy()
    
    Austevoll_Nord_df = add_num_not_Nan(Austevoll_Nord_df)
    Austevoll_Nord_df = add_num_Nan(Austevoll_Nord_df)
    Austevoll_Nord_df = add_num_bytes(Austevoll_Nord_df)
    
    an_t_1 = datetime.datetime.strptime('2022-11-23 10:40:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_2 = datetime.datetime.strptime('2023-02-28 19:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_3 = datetime.datetime.strptime('2023-06-07 07:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_4 = datetime.datetime.strptime('2023-08-29 04:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_5 = datetime.datetime.strptime('2024-05-07 06:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_6 = datetime.datetime.strptime('2024-09-05 08:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_7 = datetime.datetime.strptime('2024-11-13 20:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')


    an_t_e_1 = datetime.datetime.strptime('2022-11-23 10:40:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_e_2 = datetime.datetime.strptime('2023-02-28 19:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_e_3 = datetime.datetime.strptime('2023-06-07 12:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_e_4 = datetime.datetime.strptime('2023-08-29 09:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_e_5 = datetime.datetime.strptime('2024-05-28 12:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_e_6 = datetime.datetime.strptime('2024-09-05 08:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    an_t_e_7 = datetime.datetime.strptime('2024-11-13 20:30:00+00:00', '%Y-%m-%d %H:%M:%S%z')


    an_seg_1_e= Austevoll_Nord_df.loc[an_t_e_1:an_t_2-datetime.timedelta(seconds=1)]
    an_seg_2_e= Austevoll_Nord_df.loc[an_t_e_2:an_t_3-datetime.timedelta(seconds=1)]
    an_seg_3_e= Austevoll_Nord_df.loc[an_t_e_3:an_t_4-datetime.timedelta(seconds=1)]
    an_seg_3_e = an_seg_3_e[an_seg_3_e['System_Parameters.Input_Voltage'] <= 12.75]
    an_seg_4_e= Austevoll_Nord_df.loc[an_t_e_4:an_t_5-datetime.timedelta(seconds=1)]
    an_seg_5_e= Austevoll_Nord_df.loc[an_t_e_5:an_t_6-datetime.timedelta(seconds=1)]# maybe not
    an_seg_6_e= Austevoll_Nord_df.loc[an_t_e_6:an_t_7 - datetime.timedelta(seconds=1)] 
    an_seg_7_e = Austevoll_Nord_df.loc[an_t_e_7:None]
    an_seg_7_e = an_seg_7_e[an_seg_7_e['System_Parameters.Input_Voltage'] <= 12.5]

    hours = r_h
    if 1 in list_segments:
        an_seg_1_e = an_seg_1_e.resample(f'{hours}h').max().ffill()
    if 2 in list_segments:
        an_seg_2_e = an_seg_2_e.resample(f'{hours}h').max().ffill()
    if 3 in list_segments:
        an_seg_3_e = an_seg_3_e.resample(f'{hours}h').max().ffill()
    if 4 in list_segments:
        an_seg_4_e = an_seg_4_e.resample(f'{hours}h').max().ffill()
    if 5 in list_segments:
        an_seg_5_e = an_seg_5_e.resample(f'{hours}h').max().ffill()
    if 6 in list_segments:
        an_seg_6_e = an_seg_6_e.resample(f'{hours}h').max().ffill()
    if 7 in list_segments:
        an_seg_7_e = an_seg_7_e.resample(f'{hours}h').max().ffill()
    
    
    
    if 1 in list_segments:
        an_seg_1_e = add_time_since_t0_feature(an_seg_1_e)
        an_seg_1_e = add_year_month_week(an_seg_1_e)
        an_seg_1_e = add_austevoll_frost_temperature_feature(an_seg_1_e)
        an_seg_1_e = temp_sum_since_t0(an_seg_1_e)
        an_seg_1_e = add_fourier_feature(an_seg_1_e)
        
        
        an_seg_1_e = smooth_df_voltage_fun_df(an_seg_1_e)
        
        an_seg_1_e = add_voltage_features(an_seg_1_e)
 
        
        an_seg_1_e = an_seg_1_e[list_features]
        
    if 2 in list_segments:
        an_seg_2_e = add_time_since_t0_feature(an_seg_2_e)
        an_seg_2_e = add_year_month_week(an_seg_2_e)
        an_seg_2_e = add_austevoll_frost_temperature_feature(an_seg_2_e)
        an_seg_2_e = temp_sum_since_t0(an_seg_2_e)
        an_seg_2_e = add_fourier_feature(an_seg_2_e)
        
        
        an_seg_2_e = smooth_df_voltage_fun_df(an_seg_2_e)
        
        an_seg_2_e = add_voltage_features(an_seg_2_e)

        an_seg_2_e = an_seg_2_e[list_features]
        
    if 3 in list_segments:
        an_seg_3_e = add_time_since_t0_feature(an_seg_3_e)
        an_seg_3_e = add_year_month_week(an_seg_3_e)
        an_seg_3_e = add_austevoll_frost_temperature_feature(an_seg_3_e)
        an_seg_3_e = temp_sum_since_t0(an_seg_3_e)
        an_seg_3_e = add_fourier_feature(an_seg_3_e)
        
        
        an_seg_3_e = smooth_df_voltage_fun_df(an_seg_3_e)
        
        an_seg_3_e = add_voltage_features(an_seg_3_e)

        
        an_seg_3_e = an_seg_3_e[list_features]
        
    if 4 in list_segments:
        an_seg_4_e = add_time_since_t0_feature(an_seg_4_e)
        an_seg_4_e = add_year_month_week(an_seg_4_e)
        an_seg_4_e = add_austevoll_frost_temperature_feature(an_seg_4_e)
        an_seg_4_e = temp_sum_since_t0(an_seg_4_e)
        an_seg_4_e = add_fourier_feature(an_seg_4_e)
        
        
        an_seg_4_e = smooth_df_voltage_fun_df(an_seg_4_e)
        
        
        an_seg_4_e = add_voltage_features(an_seg_4_e)

        an_seg_4_e = an_seg_4_e[list_features]
        
    if 5 in list_segments:
        an_seg_5_e = add_time_since_t0_feature(an_seg_5_e)
        an_seg_5_e = add_year_month_week(an_seg_5_e)
        an_seg_5_e = add_austevoll_frost_temperature_feature(an_seg_5_e)
        an_seg_5_e = temp_sum_since_t0(an_seg_5_e)
        an_seg_5_e = add_fourier_feature(an_seg_5_e)
        
        
        an_seg_5_e = smooth_df_voltage_fun_df(an_seg_5_e)
        
        an_seg_5_e = add_voltage_features(an_seg_5_e)

        an_seg_5_e = an_seg_5_e[list_features]
        
    if 6 in list_segments:
        an_seg_6_e = add_time_since_t0_feature(an_seg_6_e)
        an_seg_6_e = add_year_month_week(an_seg_6_e)
        an_seg_6_e = add_austevoll_frost_temperature_feature(an_seg_6_e)
        an_seg_6_e = temp_sum_since_t0(an_seg_6_e)
        an_seg_6_e = add_fourier_feature(an_seg_6_e)
        
        
        an_seg_6_e = smooth_df_voltage_fun_df(an_seg_6_e)
        
        an_seg_6_e = add_voltage_features(an_seg_6_e)

        an_seg_6_e = an_seg_6_e[list_features]
        
    if 7 in list_segments:
        an_seg_7_e = add_time_since_t0_feature(an_seg_7_e)
        an_seg_7_e = add_year_month_week(an_seg_7_e)
        an_seg_7_e = add_austevoll_frost_temperature_feature(an_seg_7_e)
        an_seg_7_e = temp_sum_since_t0(an_seg_7_e)
        an_seg_7_e = add_fourier_feature(an_seg_7_e)
        
        
        an_seg_7_e = smooth_df_voltage_fun_df(an_seg_7_e)
        
        an_seg_7_e = add_voltage_features(an_seg_7_e)

        an_seg_7_e = an_seg_7_e[list_features]
        
    segments = {
        1: an_seg_1_e,
        2: an_seg_2_e,
        3: an_seg_3_e,
        4: an_seg_4_e,
        5: an_seg_5_e,
        6: an_seg_6_e,
        7: an_seg_7_e
    }
    
    return {key: segments[key] for key in list_segments if key in segments}