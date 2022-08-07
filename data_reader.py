#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta as td

import _utils



def get_all_input_files():
    '''
    Returns a list with all of our input files.
    (We exclude leousd bacause of a badly formed datetime) 
    '''
    return [_utils.FOLDER_data+f for f in os.listdir(_utils.FOLDER_data) if (f[-4:]=='.csv') & ('leousd_data.csv' not in f) & (f[-7:]=='usd.csv')]


def extract_df_fields(df):
    '''
    Splits the dataframe per feature and returns the required fields.
    
    Input:
        df (pd.dataframe): the original dataframe
    Output:
        (various): np.array(s) of specific features
            (Note: 'dates' is converted to datetime.datetime in this function)
    '''
    dates = np.array([datetime.fromtimestamp(d/1000) for d in df.time.values])
    #dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in df.date.values] #using Panos's processed data
    times = np.array(df.time.values)/1000 #timestamps

    idx = np.where((dates>=datetime(2020,1,1,0,0)) & (dates<datetime(2022,1,1,0,0)))[0]
    open_pr = np.array(df['open'].values)[idx]
    close_pr = np.array(df['close'].values)[idx]
    high = np.array(df['high'].values)[idx]
    low = np.array(df['low'].values)[idx]
    vol = np.array(df['volume'].values)[idx]
    return times[idx], close_pr, vol, low, high #return dates, closing price & volume


def get_sample_dates(start=datetime(2020,1,1,0,0), end=datetime(2022,1,1,0,0), interval=4):
    '''
    Returns the dates (np.array) we are after for generating the timeseries.
    'start' & 'end' indicate the first and last day of the dataset.
    'interval' 
    '''
    all_dates = []
    while start<end:
        all_dates.append(int(datetime.timestamp(start)))
        start+=td(hours=interval)
    return np.array(all_dates)


def generate_crypto_timeseries(dates, prices, volumes, lows, highs, dates_to_fill, interval):
    '''
    Given the dates and the prices of a particular crypto (vectors), this func
    generates the time-series of the price of a particular crypto. The 
    granularity of the time-series depends on the "dates_to_fill".
    
    Input:
        dates (np.array): array of dates for which we have the price of the crypto
        prices (np.array): array of the prices (1-to-1 mapping with the dates)
        dates_to_fill (np.array): the output of get_sample_dates() - vec of datetimes
        interval (int): we generate time series every 'interval' hours and interpolate
    Output:
        timeseries (np.array): the price for each point of dates_to_fill (np.nan included)
    '''

    #numHours = int(interval/2) #interpolate only if the closes value is "close enough" 
    idx = 0
    ts_price_avg = np.array([np.nan for d in dates_to_fill])
    ts_volume_avg =  np.array([np.nan for d in dates_to_fill])
    ts_low_avg =  np.array([np.nan for d in dates_to_fill])
    ts_high_avg =  np.array([np.nan for d in dates_to_fill])
    ts_price_std = np.array([np.nan for d in dates_to_fill])
    ts_volume_std =  np.array([np.nan for d in dates_to_fill])
    ts_low_std =  np.array([np.nan for d in dates_to_fill])
    ts_high_std =  np.array([np.nan for d in dates_to_fill])
    
    prev_date_to_fill = dates_to_fill[0]-4*60*60
    print(prev_date_to_fill)
    
    for date_to_fill in dates_to_fill: #for each of the exact dates we look to fill 
        if idx%360==0:
            print(datetime.fromtimestamp(date_to_fill))
                     
        rest_idx = np.where((dates>prev_date_to_fill) & (dates<=date_to_fill))[0]
        if len(rest_idx)>0:
            ts_price_avg[idx] = np.average(prices[rest_idx])  
            ts_volume_avg[idx] = np.average(volumes[rest_idx])  
            ts_low_avg[idx] = np.average(lows[rest_idx])  
            ts_high_avg[idx] = np.average(highs[rest_idx])  
            ts_price_std[idx] = np.std(prices[rest_idx])  
            ts_volume_std[idx] = np.std(volumes[rest_idx])  
            ts_low_std[idx] = np.std(lows[rest_idx])  
            ts_high_std[idx] = np.std(highs[rest_idx])  
        idx+=1
        prev_date_to_fill = date_to_fill
        '''
        diffs = np.abs(date_to_fill-dates) #all diffs
        min_diff = min(diffs) # min_diff(seconds)
        
        if min_diff<numHours*60*60: #if value is not "close enough", keep it as nan CHECK THIS
            if idx>0:
                rest_idx = np.where((dates>prev_date_to_fill) & (dates<=date_to_fill))[0]
                if len(rest_idx)>0:
                    ts_price_avg[idx] = np.average(prices[rest_idx[-1]])  
                    ts_volume_avg[idx] = np.average(volumes[rest_idx])  
                    ts_low_avg[idx] = np.average(lows[rest_idx])  
                    ts_high_avg[idx] = np.average(highs[rest_idx])  
                    ts_price_std[idx] = np.std(prices[rest_idx])  
                    ts_volume_std[idx] = np.std(volumes[rest_idx])  
                    ts_low_std[idx] = np.std(lows[rest_idx])  
                    ts_high_std[idx] = np.std(highs[rest_idx])  
            
            else: #only for the very first date, interpolate using the closest value
                price_idx = np.where(diffs==min_diff)
                ts_price_avg[idx] = np.average(prices[price_idx])  
                ts_volume_avg[idx] = np.average(volumes[price_idx])  
                ts_low_avg[idx] = np.average(lows[price_idx])  
                ts_high_avg[idx] = np.average(highs[price_idx])
                #ts_price_std[idx] = np.std(prices[rest_idx])  
                #ts_volume_std[idx] = np.std(volumes[rest_idx])  
                #ts_low_std[idx] = np.std(lows[rest_idx])  
                #ts_high_std[idx] = np.std(highs[rest_idx])  
            
        idx+=1
        prev_date_to_fill = date_to_fill
        '''
    return ts_price_avg, ts_volume_avg, ts_low_avg, ts_high_avg, ts_price_std, ts_volume_std, ts_low_std, ts_high_std
    

def process_and_save_dataset(interval=4):
    '''
    Run this once to do the preprocessing work. Saves pickles in _utils.FOLDER_timeseries.
    '''
    files = get_all_input_files()
    dates_to_fill = get_sample_dates(interval=interval)
    crypto_timeseries = dict()

    cntr = 0
    for file in files: #for each crypto file
        cntr+=1
        
        task = file.split('/')[-1].split('_')[0] #'ethusd'-like format
        df = pd.read_csv(file, encoding='ISO-8859-1') #load the raw data of the crypto
        dates, prices, volumes, lows, highs = extract_df_fields(df) # dates/features in raw data
        if len(dates)>100000: #proceed only if we have enough data (manually set)
            avg_price, avg_vol, avg_low, avg_high, std_price, std_vol, std_low, std_high = generate_crypto_timeseries(dates, prices, volumes, lows, highs, dates_to_fill, interval)
            crypto_timeseries[task] = [dates, avg_price, avg_vol, avg_low, avg_high, std_price, std_vol, std_low, std_high]
            print(cntr, '\t', task, '\t', min(dates), '\t', max(dates), '\t', len(dates))
    pickle.dump(crypto_timeseries, open(_utils.FOLDER_timeseries+'ts_'+str(interval)+'hrs_'+str(int(interval/2))+'hrs_2020-2022_avgPrice.pkl', 'wb'))
    return crypto_timeseries


def interpolate(vals):
    '''
    Given a 1d np.array(), potentially including nans, this function returns its 
    interpolated version.
    '''
    nans, l = get_nans(vals)
    vals[nans]= np.interp(l(nans), l(~nans), vals[~nans])
    return vals


def get_nans(data):
    '''Helper for interpolate()'''
    return np.isnan(data), lambda l: l.nonzero()[0]


def load_and_interpolate(interval, threshold=.995):
    '''
    Loads the preprocessed data, interpolates them and keeps only the timeseries
    that are at least "threshold" complete.
    '''
    infile = _utils.FOLDER_timeseries+'ts_'+str(interval)+'hrs_'+str(int(interval/2))+'hrs_2020-2022_avgPrice.pkl'
    data = pickle.load(open(infile, 'rb'))
    all_dates = get_sample_dates(interval=interval)
    
    data_to_use = dict() #the dict to return
    for task in data.keys():
        dates, avg_price, avg_vol, avg_low, avg_high, std_price, std_vol, std_low, std_high = data[task]
        non_nan = np.sum(~np.isnan(avg_price))  
        if non_nan/len(all_dates)>=threshold: #here we check completeness (%)
            #print(avg_vol)

            avg_price = interpolate(avg_price) # interpolation occurs here 
            avg_vol = interpolate(avg_vol)
            avg_low = interpolate(avg_low)
            avg_high = interpolate(avg_high)
            std_price = interpolate(std_price)
            std_vol = interpolate(std_vol)
            std_low = interpolate(std_low)
            std_high = interpolate(std_high)
            #print(avg_vol)
            #print('\n')
            print(task, '\t', non_nan, len(all_dates))

            data_to_use[task] = [dates, avg_price, avg_vol, avg_low, avg_high, std_price, std_vol, std_low, std_high]
    print(len(data_to_use),'cryptos are at least',100*threshold,'% complete.')
    return data_to_use
    

def form_instances(history=8, future=4, stride=2, interval=4, threshold=.995):
    '''
    Forms the instances based on the pre-specified params. Kind of a key function
    after process_and_save_dataset() has been executed once.
    
    Input:
        history: how many timesteps to use for predicting (input to lstms -or 
                                                           features in rf etc.)
        future: how many timesteps to predict (output for lstm)
        stride: how to form instances (i.e., how many timesteps to "jump" for 
                                       generating each new instance)
        interval: work on the timeseries that were generated every "interval" 
            hours per crypto (see process_and_save_dataset() and in particular
            generate_crypto_timeseries)
        threhsold: min threshold of completeness to use (exclude incomplete cryptos)
    
    Output:
        task: np.array with strings indicating the task (one per instance)
        X: np.array of shape (numInstances, history)
        y: np.array of shape (numInstances, future)
    '''
    data = load_and_interpolate(interval, threshold)
    taskids, X, y, info = [], [], [], dict()
    for task in data.keys():
        if task not in ['ustusd.csv','daiusd.csv']:
            xx, yy, task_info = form_instances_for_crypto(data[task], history, future, stride)
            X.extend(xx)
            y.extend(yy)
            taskids.extend([task for x in xx])
            info[task] = task_info
    print('\nInterval:',interval,'\tHistory',history,'\tFuture:',future,'\tStride:', stride)
    print(len(taskids), 'instances were formed overall (', len(taskids)/len(data), 'per crypto )')
    return np.array(taskids), np.array(X), np.array(y), info

        
def form_instances_for_crypto(vals, history, future, stride):
    '''
    Forms the instances for a single crypto, with the specified params.
    Called within form_all_instnces().
    
    Input:
        vals: the time series (1d) of a single crypto
        history: how many timesteps to use for predicting (input to lstms -or 
                                                           features in rf etc.)
        future: how many timesteps to predict (output for lstm)
        stride: how to form instances (i.e., how many timesteps to "jump" for 
                                       generating each new instance)
    
    Output:
        x: list with the generated input for this crypto (numInstances, history)
        y: list with the generated outpiut for this crypto (numInstances, future)
    '''
    dates, avg_price, avg_vol, avg_low, avg_high, std_price, std_vol, std_low, std_high = vals
    maxi, mini = max(avg_price), min(avg_price)
    avg_price = [(a-mini)/(maxi-mini) for a in avg_price]
    avg_vol = [(a-mini)/(maxi-mini) for a in avg_vol]
    std_price = [(a-mini)/(maxi-mini) for a in std_price]
    std_vol = [(a-mini)/(maxi-mini) for a in std_vol]

    xx, y = [], []
    for i in range(0, len(avg_price), stride):
        if i+history+future<=len(vals[1]): #we pass
            x = []
            for j in range(i, i+history):
                x.append([avg_price[j], avg_vol[j], std_price[j], std_vol[j]])
            xx.append(x)
            #x.append([avg_price[i:i+history], avg_vol[i:i+history], std_price[i:i+history], std_vol[i:i+history]])#, std_low[i:i+history], std_high[i:i+history]])
            y.append(avg_price[i+history:i+history+future])
    
    return xx, y, [maxi, mini]
