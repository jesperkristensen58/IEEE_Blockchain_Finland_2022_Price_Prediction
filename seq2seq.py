#!/usr/bin/env python3
# -*- coding: utf-8 -*-
seed_value= 0
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)

import time
import pickle

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
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
from keras import optimizers

import _utils
from data_reader import form_instances


def get_train_test_data(history, future, stride, interval, threshold, test_split):
    '''Returns the train.test data, along with the info about their min/max scaling'''
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
        
        trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], 1)
        testY = testY.reshape(testY.shape[0], testY.shape[1], 1)
            
        train[taskid] = [trainX, trainY]
        test[taskid] = [testX, testY]
    return train, test, info


def shift_data(X):
    '''Shifts the data by one timestep'''
    zero_pad = np.zeros((1,_utils.MODELLING_numFeatures))
    X_2 = np.empty([X.shape[0], X.shape[1], X.shape[2]])
    for i in range(len(X)):
        X_2[i] = np.vstack([zero_pad, X[i,:-1,:]])       
    return X_2


def define_models(n_units, n_input, n_output, input_ts, output_ts):
    '''Definition of encoder and decoder models'''
    encoder_inputs = Input(shape=(None, n_input), name='encoder_input')
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(None, n_output), name='decoder_input')
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(n_output))
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model


def train_all_models(n_units, test_split=0.2):
    '''Trains all models and returns them (called from train_seq2seq)'''
    which = ['omgusd.csv', 'xlmusd.csv', 'xtzusd.csv', 'dshusd.csv', 'batusd.csv', 'trxusd.csv', 'zrxusd.csv', 'bsvusd.csv', 'iotusd.csv', 'ltcusd.csv', 'leousd.csv']
    train, test, info = get_train_test_data(_utils.MODELLING_past, _utils.MODELLING_future,  _utils.MODELLING_stride, _utils.MODELLING_interval, _utils.MODELLING_threshold, test_split)     
    
    models, test_data = dict(), dict()
    for task in train.keys():
        if task in which: #add/remove a "not" to run in parallel
            trainX, trainY = train[task]        
            testX, testY = test[task]
            task_info = info[task]
        
            val_split_threshold = int(0.8*len(trainX))
            valX, valY = trainX[val_split_threshold:], trainY[val_split_threshold:]
            trainX, trainY = trainX[:val_split_threshold], trainY[:val_split_threshold]
                        
            trainX_shifted = shift_data(trainY)
            valX_shifted = shift_data(valY)
            trainX_shifted[:,0,:] = trainX[:,-1,:] #here I add the last value of the encoder input as the first value of the decoder input
            valX_shifted[:,0,:] = valX[:,-1,:]
    
            model = define_models(n_units, _utils.MODELLING_numFeatures, _utils.MODELLING_numFeatures, _utils.MODELLING_past, _utils.MODELLING_future) 
            adam = optimizers.Adam(lr=0.0001)
            
            model.compile(loss='mean_squared_error', metrics=['mse'], optimizer=adam)
            es = EarlyStopping(patience=100, monitor="val_loss", mode='min', restore_best_weights=True)#or val_loss?
            
            start = time.time()
            model.fit([trainX, trainX_shifted], trainY, validation_data=([valX,valX_shifted], valY),
                      batch_size=8,epochs=1000,shuffle=False,verbose=0, callbacks=[es])
            end = time.time()

            outfile = 'seq2seq'+task[:-4]+'_h'+str(_utils.MODELLING_past)+'f'+str(_utils.MODELLING_future)+'s'+str(_utils.MODELLING_stride)+'i'+str(_utils.MODELLING_interval)+'feat'+str(_utils.MODELLING_numFeatures)+'.p'
            pickle.dump(model, open(_utils.FOLDER_models+outfile, 'wb'))
            
            
            models[task] = model
            test_data[task] = [testX, testY, task_info, [start, end]]
    return models, test_data


def predict_sequence(enc, dec, data, n_ts, n_feat):
    first_val_for_decoder = data[-1,:]
    data = np.reshape(data, (1, data.shape[0], n_feat))
	
    state = enc.predict(data) 
    target_seq = np.array([first_val_for_decoder[_] for _ in range(n_feat)]).reshape(1, 1, n_feat)    
    output = []
    for t in range(n_ts):		
        yhat, h, c = dec.predict([target_seq] + state) 
        output.append(yhat[0,0,:]) 
        state = [h, c]	
        target_seq = yhat
    return(np.array(output))  
    

def train_seq2seq(n_units):
    '''Runs everyting & stores the predictions/models'''
    models, test_data = train_all_models(n_units)
    for task in models.keys():
        model = models[task]
        testX, testY, task_info, times = test_data[task]
 
        encoder_inputs = model.input[0]   
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output 
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)
        
        decoder_inputs = model.input[1] 
        decoder_state_input_h = Input(shape=(n_units,), name='input_3')
        decoder_state_input_c = Input(shape=(n_units,), name='input_4')
        
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        preds = []
        for i in range(testX.shape[0]):
            decoder_prediction = predict_sequence(encoder_model, decoder_model, testX[i], n_steps=testY.shape[1], num_feat=_utils.MODELLING_numFeatures) 
            preds.append(decoder_prediction)
        preds = np.array(preds)
        
        outfile = 'seq2seq'+task[:-4]+'_h'+str(_utils.MODELLING_past)+'f'+str(_utils.MODELLING_future)+'s'+str(_utils.MODELLING_stride)+'i'+str(_utils.MODELLING_interval)+'feat'+str(_utils.MODELLING_numFeatures)+'.p'
        pickle.dump([testY, preds, task_info, times], open(_utils.FOLDER_predictions+outfile, 'wb'))  
        
        mape = 100*(np.abs(testY-preds)/testY)
        print(task,'\t', np.average(np.average(mape, axis=0)), '\t', np.average(mape, axis=0))