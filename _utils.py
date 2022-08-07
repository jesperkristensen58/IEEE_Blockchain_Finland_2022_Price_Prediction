#!/usr/bin/env python3
# -*- coding: utf-8 -*-
FOLDER_data = 'crypto_dataset/'#Download from https://www.kaggle.com/tencars/392-crypto-currency-pairs-at-minute-resolution
FOLDER_models = 'new_models/'
FOLDER_predictions = 'new_preds/'
FOLDER_timeseries = 'input_data/'

MODELLING_interval = 8 #time window (timestep) of our modelling: each k hours
MODELLING_threshold = .995
MODELLING_stride = 1 #how many timesteps to stride for generating instances
MODELLING_past = 12 #how many intervals to use as input
MODELLING_future = 3 #how many intervals to predict
MODELLING_numFeatures = 1

DATA_coins = ['omgusd.csv','xlmusd.csv','xtzusd.csv','dshusd.csv','batusd.csv','trxusd.csv','zrxusd.csv','bsvusd.csv','iotusd.csv','ltcusd.csv','leousd.csv','uosusd.csv','eosusd.csv','btcusd.csv','xrpusd.csv','vsyusd.csv','xmrusd.csv','zecusd.csv','ethusd.csv','etpusd.csv','neousd.csv','etcusd.csv']


PARAMS_rf = {'n_estimators':[50,100,250,500]}
PARAMS_lstm = {'num_units':[32,128,512],
               'lr':[.0001, .00001],
               'dout':[.2]} 
MODEL_PARAMS = {'rf':PARAMS_rf, 'lstm':PARAMS_lstm}
