#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import time
from sklearn.ensemble import RandomForestRegressor

from data_reader import form_instances
import _utils


seed_value= 0
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.set_random_seed(seed_value)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Dense, RepeatVector, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


def limit_num_targets(trainY, keep_every_k_hours=24):
    real_trainY = []
    for i in range(len(trainY)):
        target = []
        for hour in range(trainY.shape[1]):
            if (hour%(keep_every_k_hours-1)==0) & (hour>0):
                target.append(trainY[i][hour])
        real_trainY.append(target)
    return np.array(real_trainY)


def get_train_test_data(history, future, stride, interval, threshold, test_split):
    all_taskids, all_X, all_y, info = form_instances(history, future, stride, interval, threshold)    
    train, test = dict(), dict()
    for taskid in set(all_taskids):
        
        idx = np.where(all_taskids==taskid)[0]
        train_limit = int(len(idx)*(1-test_split))
        
        train_idx, test_idx = idx[:train_limit], idx[train_limit:]  
        
        trainX, trainY = all_X[train_idx], all_y[train_idx]
        testX, testY = all_X[test_idx], all_y[test_idx]
        
        trainX = trainX[:,:,:_utils.MODELLING_numFeatures]
        testX = testX[:,:,:_utils.MODELLING_numFeatures]
        
        train[taskid] = [trainX, trainY]
        test[taskid] = [testX, testY]
        
    return train, test, info


def get_normalised_data(train, test):
    mu = np.average(train, axis=0)
    std = np.std(train, axis=0)
    return (train-mu)/std, (test-mu)/std, {'mu':mu, 'std':std}

def get_scaled_data(train, test):
    mini = np.min(train, axis=0)
    maxi = np.max(train, axis=0)
    return (train-mini)/(maxi-mini), (test-mini)/(maxi-mini)


def train_model(modelname, multistep=False, test_split=0.2):
    history = _utils.MODELLING_past
    future = _utils.MODELLING_future
    stride = _utils.MODELLING_stride
    interval = _utils.MODELLING_interval
    threshold = _utils.MODELLING_threshold
    num_features = _utils.MODELLING_numFeatures #whether to use only the price (1) or the 4 features (4)

    train, test, info = get_train_test_data(history, future, stride, interval, threshold, test_split)
    for task in train.keys():
        trainX, trainY = train[task]        
        testX, testY = test[task]
        task_info = info[task]
        
        #normalise input and target
        if modelname not in ['last', 'lastweek']:
            if multistep==False:
                trainY, testY = trainY[:,0].reshape(-1,1), testY[:,0].reshape(-1,1) #single task (only first timestep prediction) or multi-step
            else:
                if modelname=='lstm':
                    trainY, testY = trainY.reshape(trainY.shape[0], trainY.shape[1],1), testY.reshape(testY.shape[0], testY.shape[1], 1)

        #other preprocessing bits: if sklearn/mlp, convert temporal data to features
        if modelname not in ['lstm', 'last', 'lastweek']: 
            trainX = trainX.reshape(trainX.shape[0], -1)
            testX = testX.reshape(testX.shape[0], -1)
    
        #split train/val
        val_split_threshold = int(0.8*len(trainX))
        valX, valY = trainX[val_split_threshold:], trainY[val_split_threshold:]
        trainX, trainY = trainX[:val_split_threshold], trainY[:val_split_threshold]
                
        print(trainX.shape, trainY.shape)
        #models
        if modelname=='lastweek':
            _preds = []
            for i in range(len(testY)):
                #lastVal = testX[i,-_utils.MODELLING_future:,0]
                lastVal = list(testX[i,-3:,0])
                for kk in range(6):
                    lastVal.extend(list(testX[i,-3:,0]))
                _preds.append(lastVal)
                
            _preds = np.array(_preds)
            outfile = modelname+task[:-4]+'_h'+str(_utils.MODELLING_past)+'f'+str(_utils.MODELLING_future)+'s'+str(_utils.MODELLING_stride)+'i'+str(_utils.MODELLING_interval)+'feat'+str(_utils.MODELLING_numFeatures)+'.p'

            pickle.dump([testY, _preds, task_info, 0], open(outfile, 'wb'))
        elif modelname=='last':
            _preds = []
            for i in range(len(testY)):
                lastVal = testX[i][-1][0]
                this_instance_vals = []
                for k in range(testY.shape[1]):
                    this_instance_vals.append(lastVal)
                _preds.append(this_instance_vals)
            _preds = np.array(_preds)
            
        elif modelname in ['lasso', 'lr', 'rf']:
            model = get_model(modelname)
            start = time.time()
            model.fit(trainX, trainY)
            end = time.time()
            
        elif modelname in ['mlp', 'lstm']:            
            model = get_model(modelname)
            es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
            model.compile(loss="mean_squared_error",  optimizer=Adam(lr=0.0001), metrics=['mse'])
            
            start = time.time()
            model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=8, epochs=1000, shuffle=False,verbose=0, callbacks=[es])
            end = time.time()
            
        #save best model and predictions on test set
        if modelname not in ['last', 'lastweek']:
            outfile = modelname+task[:-4]+'_h'+str(_utils.MODELLING_past)+'f'+str(_utils.MODELLING_future)+'s'+str(_utils.MODELLING_stride)+'i'+str(_utils.MODELLING_interval)+'feat'+str(_utils.MODELLING_numFeatures)+'.p'
            _preds = model.predict(testX)
            if modelname in ['lstm', 'mlp']:
                pickle.dump(model, open(_utils.FOLDER_models+outfile, 'wb'))        
                pickle.dump([testY, _preds, task_info, [start,end]], open(_utils.FOLDER_predictions+outfile, 'wb'))  
                del model
            elif modelname in ['rf']:
                pickle.dump([testY, _preds, task_info, [start,end]], open(_utils.FOLDER_predictions+outfile, 'wb'))  
        
            
            mape = 100*(np.abs(testY-_preds)/testY)
            print(task,'\t', np.average(np.average(mape, axis=0)), '\t', start, '\t', end)


def get_model(modelname):
    if modelname=='rf':
        model = RandomForestRegressor(n_estimators=200, verbose=0)
    elif modelname=='lstm':
        model = get_lstm_model_multi()
    elif modelname=='mlp':
        model = get_mlp_model()
    return model


def get_lstm_model_multi():
    # Define hyperparams
    num_units1 = 32
    
    # Define some lstm-based model of 2 layers + 1 dense for the prediction
    inputs = Input(shape=(_utils.MODELLING_past, _utils.MODELLING_numFeatures))
    x = LSTM(num_units1, return_sequences=False)(inputs)
    x = RepeatVector(int(_utils.MODELLING_future))(x)
    x = LSTM(num_units1, return_sequences=True)(x)
    output = TimeDistributed(Dense(1))(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def get_mlp_model():
    # Define hyperparams
    num_units1 = 32
    
    # Define some lstm-based model of 2 layers + 1 dense for the prediction
    inputs = Input(shape=(_utils.MODELLING_past*_utils.MODELLING_numFeatures,))
    x = Dense(num_units1, activation='linear')(inputs)#, kernel_regularizer=l2(0.25))(inputs)
    x = Dense(num_units1, activation='linear')(x)#, kernel_regularizer=l2(0.25))(inputs)
    #x = Dropout(pct_dout)(x)
    output = Dense(_utils.MODELLING_future)(x)
        
    model = Model(inputs=inputs, outputs=output)
    return model    