#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from datetime import datetime
import pickle

import _utils


def get_multitask_results():
    coins = _utils.DATA_coins
    
    infile = _utils.FOLDER_predictions+'seq2seq16_multires_h12f3s1i8feat1.p'
    actual, preds, tasks, [start, end] = pickle.load(open(infile, 'rb'))
    tasks = np.array(tasks)
    mapes, r2s, mses = [], [], []
    for coin in coins:
        idx = np.where(tasks==coin)[0][0]
        task = tasks[idx][:-4]

        ac = actual[:,:,idx]
        pr = preds[:,:,idx]
        
        mape_per_ts = get_mape(ac, pr)
        r2_per_ts = get_r2(ac, pr)
        mse_per_ts = get_mse(ac, pr)
        
        mapes.append(np.average(mape_per_ts))
        r2s.append(np.average(r2_per_ts))
        mses.append(np.average(mse_per_ts))
        
        print(str(np.round(100*np.average(mse_per_ts),4)))
        #print(task[:-3].upper())
    print(np.round(100*np.median(mses),4))
    
    
def get_all_results(modelname):
    coins = _utils.DATA_coins
    coins.sort()
    mapes, r2s, mses = [], [], []
    for coin in coins:
        coin_str = coin[:3]
        infile = _utils.FOLDER_predictions+modelname+coin[:-4]+'_h12f3s1i8feat1.p'  #for lstm we want the 16-unit version (ie, without 32 in the filename)
        actual, preds, [maxi, mini], times = pickle.load(open(infile, 'rb'))
        if len(actual.shape)==3:
            actual = actual[:,:,0]
            preds = preds[:,:,0]
        '''
        new_actual, new_preds = np.zeros(actual.shape), np.zeros(preds.shape)
        for i in range(3):
            new_actual[:,i] = (actual[:,i]*(maxi-mini))+mini
            new_preds[:,i] = (preds[:,i]*(maxi-mini))+mini
        #plot_predictions(new_actual, new_preds, coin)
        '''
        
        mape_per_ts = get_mape(actual, preds)
        r2_per_ts = get_r2(actual, preds)
        mse_per_ts = get_mse(actual, preds)
        
        mapes.append(np.average(mape_per_ts))
        r2s.append(np.average(r2_per_ts))
        mses.append(np.average(mse_per_ts))
        
        print(str(np.round(100*np.average(mapes),4)))#, '\t', 100*np.average(mape_per_ts), '\t', np.average(np.sqrt(mse)))
    print(np.round(100*np.median(mses),4))
    
    
    
def analyse_predictions_mt_vs_s2s():
    coins = _utils.DATA_coins
    
    infile = _utils.FOLDER_predictions+'seq2seq16_multires_h12f3s1i8feat1.p'
    mt_actual, mt_preds, mt_tasks, _ = pickle.load(open(infile, 'rb'))
    mt_tasks = np.array(mt_tasks)
    mapes, r2s, mses = [], [], []
    for coin in coins:
        idx = np.where(mt_tasks==coin)[0][0]

        ac, pr = mt_actual[:,:,idx],mt_preds[:,:,idx]
        
        mape_mt = np.average(get_mape(ac, pr))
        
        infile = _utils.FOLDER_predictions+'seq2seq'+coin[:-4]+'_h12f3s1i8feat1.p'  #for lstm we want the 16-unit version (ie, without 32 in the filename)
        actual, preds, [maxi, mini], times = pickle.load(open(infile, 'rb'))
        if len(actual.shape)==3:
            actual = actual[:,:,0]
            preds = preds[:,:,0]
        mape_s2s = np.average(get_mape(actual, preds))
        
        relative_gain = (mape_s2s-mape_mt)/mape_s2s
        stdev = np.std([actual[i][0] for i in range(len(actual))])
        resid = [0]
        for i in range(1, len(actual)):
            resid.append(actual[i][0]-actual[i-1][0])
        print(coin, '\t', relative_gain)



def get_model_predictions(modelname, coin):
    actual, preds, info, times =  pickle.load(open(_utils.FOLDER_predictions+modelname+coin[:-4]+'_h12f3s1i8feat1.p' , 'rb'))
    if len(actual.shape)==3:
        actual = actual[:,:,0]
        preds= preds[:,:,0]
    return actual, preds, info, times



def get_data_for_mape_chart(metric):
    #First process the multitask one
    infile = _utils.FOLDER_predictions+'seq2seq16_multires_h12f3s1i8feat1.p'
    actual_raw, preds_raw, tasks, _ = pickle.load(open(infile, 'rb'))
    #now rearrange the coins so that they are in the same order with the rest
    tasks = np.array(tasks)
    actual, preds = [], []
    for coin in _utils.DATA_coins:
        idx = np.where(tasks==coin)[0][0]
        actual.append(actual_raw[:,:,idx])
        preds.append(preds_raw[:,:,idx])
    actuals_multitask = np.array(actual)
    preds_multitask = np.array(preds)
    print(actuals_multitask.shape)
        
    mapes_multitask, mapes_lstm, mapes_s2s, mapes_mlp = [], [], [], []
    for i in range(len(_utils.DATA_coins)):
        coin = _utils.DATA_coins[i]
        actuals_mlp, preds_mlp, _, times_mlp = get_model_predictions('mlp', coin)
        actuals_lstm, preds_lstm , _, times_lstm = get_model_predictions('lstm', coin)
        actuals_s2s, preds_s2s, _, times_s2s = get_model_predictions('seq2seq', coin)
        
        if metric=='MAPE':
            mapes_mlp.append(get_mape(actuals_mlp, preds_mlp))
            mapes_lstm.append(get_mape(actuals_lstm, preds_lstm))
            mapes_s2s.append(get_mape(actuals_s2s, preds_s2s))
            mapes_multitask.append(get_mape(actuals_multitask[i], preds_multitask[i]))
        elif metric=='r^2':
            mapes_mlp.append(get_r2(actuals_mlp, preds_mlp))
            mapes_lstm.append(get_r2(actuals_lstm, preds_lstm))
            mapes_s2s.append(get_r2(actuals_s2s, preds_s2s))
            mapes_multitask.append(get_r2(actuals_multitask[i], preds_multitask[i]))
    return np.array(mapes_mlp), np.array(mapes_lstm), np.array(mapes_s2s), np.array(mapes_multitask)


def plot_metric_per_timestep(metric):
    ffnn, lstm, s2s, multitask = get_data_for_mape_chart(metric)
    x = np.arange(3)+1
    plt.clf()
    plt.plot(x, np.average(lstm, axis=0), label='LSTM', linestyle='dashed', marker='o')
    plt.plot(x, np.average(s2s, axis=0), label='s2s', linestyle='dotted', marker='o')
    plt.plot(x, np.average(multitask, axis=0), label='MTs2s', linestyle='dashdot', marker='o')
    plt.legend(fontsize=18)
    plt.title('Average '+str(metric)+' per timestep', fontsize=22)
    plt.xlabel('Future timestep', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.grid(alpha=0.1)
    plt.xticks(np.arange(3)+1, fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('plots/'+metric+'_per_ts.png', dpi=300, bbox_inches='tight')


def plot_training_times():
    infile = _utils.FOLDER_predictions+'seq2seq16_multires_h12f3s1i8feat1.p'
    _, _, _, [mt_start, mt_end] = pickle.load(open(infile, 'rb'))
    mt_duration = (datetime.fromtimestamp(mt_end)-datetime.fromtimestamp(mt_start)).seconds/60.0
    
    vals = []
    
    coins = _utils.DATA_coins
    for modelname in ['mlp', 'lstm', 'seq2seq']:
        durations = []
        for coin in coins:
            infile = _utils.FOLDER_predictions+modelname+coin[:-4]+'_h12f3s1i8feat1.p'  #for lstm we want the 16-unit version (ie, without 32 in the filename)
            actual, preds, [maxi, mini], times = pickle.load(open(infile, 'rb'))
            diff = (datetime.fromtimestamp(times[1])-datetime.fromtimestamp(times[0])).seconds/60.0
            durations.append(diff)
        vals.append(np.average(durations))
    vals.append(mt_duration)
    print(vals)     
    names = ['FF', 'LSTM', 's2s', 'MTs2s']
    colours = ['cyan', 'maroon', 'orange', 'violet']
    
    # Figure Size
    fig, ax = plt.subplots(figsize =(4, 3))
    
    # Horizontal Bar Plot
    for i in range(len(names)):
        ax.bar(names[i], vals[i],color=colours[i])
    plt.ylabel('Time (mins)', fontsize=14)
    plt.savefig('boxplot.png', dpi=250, bbox_inches='tight')


def analyse_future_multitask():
    mapes, naive = [], []
    for future in [21]:
        infile = _utils.FOLDER_predictions+'seq2seq16_multires_h12f'+str(future)+'s1i8feat1.p'
        actual, preds, coins, _ = pickle.load(open(infile, 'rb'))
        f_mapes, n_mapes = [], []
        for i in range(len(coins)):
            ac, pr = actual[:,:,i], preds[:,:,i]
            mape = get_mape(ac, pr)
            f_mapes.append(mape)
            
            naive_infile = _utils.FOLDER_predictions+'lastweek'+coins[i][:-4]+'_h12f'+str(future)+'s1i8feat1.p'
            ac, pr, _, _ = pickle.load(open(naive_infile, 'rb'))
            print(ac.shape, pr.shape)
            mape = get_mape(ac, pr)
            n_mapes.append(mape)
        mapes.append(f_mapes)
        naive.append(n_mapes)
    return mapes, naive


def plot_mts2s_preds_vs_actual(crypto):
    width = 0.5
    timestep = 0
    
    infile = _utils.FOLDER_predictions+'seq2seq16_multires_h12f3s1i8feat1.p'
    actual, preds, tasks, __ = pickle.load(open(infile, 'rb'))
    tasks = np.array(tasks)
    idx = np.where(tasks==crypto+'usd.csv')[0][0]
    actual, preds = actual[:,timestep,idx], preds[:,timestep,idx]
    plt.clf()
    x = np.arange(len(actual))
    plt.plot(x, actual, linewidth=width*2, label='actual')
    plt.plot(x, preds, linewidth=width*3, linestyle='dotted', label='MTs2s')
    
    
    infile = _utils.FOLDER_predictions+'seq2seq'+crypto+'usd_h12f3s1i8feat1.p'
    actual2, preds2, _, __ = pickle.load(open(infile, 'rb'))
    actual2, preds2 = actual2[:,timestep,0], preds2[:,timestep,0]

    plt.plot(x, preds2, linewidth=width, linestyle='dotted', label='s2s')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.1)
    plt.title('Predicted vs Actual Prices for '+str(crypto.upper()))
    plt.savefig(crypto+'.png', dpi=300, bbox_inches='tight')
    
    mape1 = np.average(np.abs(actual-preds)/actual)
    mape2 = np.average(np.abs(actual-preds2)/actual)
    print(mape1, mape2)


def plot_future_analysis():
    mapes, naive = analyse_future_multitask()
    plt.clf()
    ax = plt.figure().gca()

    i=0
    y1 = np.average(mapes[i], axis=0)
    y2 = np.average(naive[i], axis=0)
    x =np.arange(len(y1))+1
    
    plt.plot(x,y1, label='MTs2s', linestyle='dashed', marker='o')
    plt.plot(x,y2, label='LV', linestyle='dotted', marker='o')
    plt.grid(alpha=0.1)
    plt.xlabel('Future timestep', fontsize=18)
    plt.ylabel('MAPE', fontsize=18)
    plt.legend(fontsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig('future_analysis.png', dpi=300, bbox_inches='tight')
        
            
            
def analyse_history_multitask():
    mapes = []
    for past in [3,6,9,12,15,18,21]:
        infile = _utils.FOLDER_predictions+'seq2seq16_multires_h'+str(past)+'f3s1i8feat1.p'
        actual, preds, coins, _ = pickle.load(open(infile, 'rb'))
        f_mapes = []
        for i in range(len(coins)):
            ac, pr = actual[:,:,i], preds[:,:,i]
            mape = get_mape(ac, pr)
            f_mapes.append(mape)
        mapes.append(f_mapes)
    return mapes
    
    
def get_mape(actual, pred):
    all_errs = []
    for i in range(len(actual)):
        ts_err = []
        for ts in range(actual.shape[1]):
            ac = actual[i,ts]
            pr = pred[i,ts]
            diff = np.abs(ac-pr)/ac
            ts_err.append(diff)
        all_errs.append(ts_err)
    return np.average(np.array(all_errs), axis=0)


def get_mse(actual, pred):
    all_errs = []
    for i in range(len(actual)):
        ts_err = []
        for ts in range(actual.shape[1]):
            ac = actual[i,ts]
            pr = pred[i,ts]
            diff = (ac-pr)**2
            ts_err.append(diff)
        all_errs.append(ts_err)
    return np.average(np.array(all_errs), axis=0)


def get_r2(actual, pred):
    r2s = []
    for ts in range(actual.shape[1]):
        ac = actual[:,ts]
        pr = pred[:,ts]
        avg_ac = np.average(ac)
        r2 = 1-(np.sum((pr-ac)**2))/(np.sum((ac-avg_ac)**2))
        r2s.append(r2)
    return np.array(r2s)
    

def get_smape(actual, pred):
    return np.array([np.abs(actual[i]-pred[i])/((actual[i]+pred[i])/2) for i in range(len(actual))])


def get_mae(actual, pred):
    return np.array([np.abs(actual[i]-pred[i]) for i in range(len(actual))])


def get_mda(actual, pred):
    ac_diff = np.zeros((len(actual),actual.shape[1]-1))
    pr_diff = np.zeros((len(pred),actual.shape[1]-1))
    
    for i in range(len(actual)):
        for hour in range(1,len(actual[i])):
            d_a = actual[i][hour]-actual[i][hour-1]
            d_p = pred[i][hour]-pred[i][hour-1]
            if d_a>0:
                ac_diff[i,hour-1] = 1
            if d_p>0:
                pr_diff[i,hour-1] = 1
    correct = []
    for i in range(len(ac_diff)):
        instance_correct = []
        for hour in range(len(ac_diff[i])):
            if ac_diff[i][hour]==pr_diff[i][hour]:
                instance_correct.append(1)
            else:
                instance_correct.append(0)
        correct.append(instance_correct)
    return np.array(correct)


def plot_predictions(actual, pred, name):
    plt.clf()
    ac = actual[:,2]
    pr = pred[:,2]
    x = np.arange(len(ac))
    plt.plot(x, ac)
    plt.plot(x, pr)
    plt.savefig('plots/'+str(name)+'_.png', dpi=200)
    