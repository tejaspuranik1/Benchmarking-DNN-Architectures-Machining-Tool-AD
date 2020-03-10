# -*- coding: utf-8 -*-
"""
@author: Tejas Puranik
This file contains the code for performing benchmarking of deep neural network architectures
"""

#%% Importing packages and defining required functions

from datetime import datetime
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import Functions_Per_Second_Approach
import eli5
from eli5.sklearn import PermutationImportance
from keras.models import Model
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels
from keras.metrics import binary_crossentropy

import time
#%% Loading the experiment data and identifying appropriate labels

path1 = '_data_20190502.csv'
path2 = '_data_20190506.csv'

df1 = pd.read_csv(path1, header=0)
df1 = df1.iloc[0:1946, 1:]  # For some reason an unnamed empty column is present so dropping it. Can fix code later

df2 = pd.read_csv(path2, header=0)
df2 = df2.iloc[:, 1:]

# Use only data with surface speed = 150 M/min
df1 = df1[df1['Surface Speed (M/min)'] == 150]
df2 = df2[df2['Surface Speed (M/min)'] == 150]

df = df2  # Using this data for training and validation
df_ts = df1  # Use this data for testing

labels_Tool = df['Tool']
labels_Tool_ts = df_ts['Tool']

#%% DOE Definition

DOE = pd.read_csv('DOE.csv')

m1 = {1: 'Turning', 2: 'Facing', 3: 'Both'}
# m2 = {}
m3 = {1: 500}
# m4 = {1: 0.025, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.3, 6: 0.5, 7: 0.75, 8: 0.95}
m4 = {1: 0.05, 2: 0.2, 3: 0.4, 4: 0.8}
m5 = {1: 'adam'}
# m6 = {1: 2, 2: 4, 3: 6}
m6 = {1: 2, 2: 4, 3: 6}
m7 = {1: 4, 2: 6, 3: 8}

DOE['Operation'] = DOE['Operation'].replace(m1)
DOE['Epochs'] = DOE['Epochs'].replace(m3)
DOE['Batch Size'] = DOE['Batch Size'].replace(m4)
DOE['Optimizer'] = DOE['Optimizer'].replace(m5)
DOE['Hidden Layers'] = DOE['Hidden Layers'].replace(m6)
DOE['Hidden Unit Multiplier'] = DOE['Hidden Unit Multiplier'].replace(m7)

#%% Identifying feature subsets: Rename col_names to whichever you want to use
col_names_all = ['Current Mean', 'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS',
                 'Current Crest Factor', 'Current Peak', 'Vibration Mean', 'Vibration Std', 'Vibration Skewness',
                 'Vibration Kurtosis','Vibration RMS', 'Vibration Crest Factor', 'Vibration Peak',
                 'LS1cmd', 'LS1load', 'LS1speed', 'LX1load', 'LX1actw', 'LZ1actw', 'LZ1load', 'x', 'y', 'z',
                 'Cutting Depth (mm)', 'Finishing Feed Rate (mm/rev)', 'Surface Speed (M/min)']

col_names_sensor = ['Current Mean', 'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS',
                    'Current Crest Factor', 'Current Peak', 'Vibration Mean', 'Vibration Std', 'Vibration Skewness',
                    'Vibration Kurtosis','Vibration RMS', 'Vibration Crest Factor', 'Vibration Peak']

col_names_sensor_plusdoe = ['Current Mean', 'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS',
                            'Current Crest Factor', 'Current Peak', 'Vibration Mean', 'Vibration Std',
                            'Vibration Skewness', 'Vibration Kurtosis','Vibration RMS', 'Vibration Crest Factor',
                            'Vibration Peak', 'Cutting Depth (mm)', 'Finishing Feed Rate (mm/rev)',
                            'Surface Speed (M/min)']

col_names_ctrl_plusdoe = ['LS1cmd', 'LS1load', 'LS1speed', 'LX1load', 'LX1actw', 'LZ1actw', 'LZ1load',
                          'Cutting Depth (mm)', 'Finishing Feed Rate (mm/rev)', 'Surface Speed (M/min)']

col_names_ctrl_nodoe = ['LS1cmd', 'LS1load', 'LS1speed', 'LX1load', 'LX1actw', 'LZ1actw', 'LZ1load',
                        'x', 'z']

col_names_ctrl_nodoepos = ['LS1cmd', 'LS1load', 'LS1speed', 'LX1load', 'LX1actw', 'LZ1actw', 'LZ1load']

# col_names = col_names_ctrl_nodoe

col_set = {1: col_names_sensor, 2: col_names_sensor_plusdoe, 3: col_names_ctrl_nodoepos, 4: col_names_ctrl_nodoe,
           5: col_names_ctrl_plusdoe, 6: col_names_all}

outs = pd.DataFrame(columns=['Bal. Acc Train', 'Bal. Acc Val', 'Bal. Acc Test', 'Bal Acc Sample Train',
                             'Bal Acc Sample Val', 'Bal Acc Sample Test', 'Conf. Train', 'Conf. Val',
                             'Conf. Test', 'Training ', 'Training Time Normalized', 'Testing Time',
                             'Testing Time Normalized', 'AE Bal. Acc Train', 'AE Bal. Acc Val', 'AE Bal. Acc Test',
                             'AE Conf. Train', 'AE Conf. Val', 'AE Conf. Test'])

#%% Execute DOE 
# Either as a single case or in batch mode

single_case = False
if single_case:
    range_doe = [35] # Choose which single case you want to run
else:
    range_doe = range(DOE.shape[0])
    
for i in range_doe:  #
    print('\nOn DOE Case', i, 'of', DOE.shape[0])
    current_run = DOE.iloc[i, :]
    # Training and Testing split
    df = df2  # Using this for training and other for validation
    df_ts = df1  # Use the other one for testing

    if current_run['Operation'] == 'Both':
        boo = df.Operation == df.Operation
        boo_ts = df_ts.Operation == df_ts.Operation
    else:
        boo = df.Operation == current_run['Operation']
        boo_ts = df_ts.Operation == current_run['Operation']

    df = df[boo]
    df_ts = df_ts[boo_ts]

    labels_Tool = df['Tool']
    labels_Tool_ts = df_ts['Tool']

    col_names = col_set[current_run['Parameter Set']]
    importlib.reload(Functions_Per_Second_Approach)

    d = {'Good': 1, 'Bad': 0}
    labels = labels_Tool.map(d)
    y_test = df_ts['Tool'].map(d)
    x_train, x_valid, y_train, y_valid = train_test_split(df, labels, test_size=0.25, shuffle=True)

    df_3_tr = x_train[col_names]
    df_3_val = x_valid[col_names]
    df_3_ts = df_ts[col_names]

    #%% Deep Neural Network
    model_d = Functions_Per_Second_Approach.deep_fully_connected(df_3_tr, current_run['Hidden Layers'],
                                                 current_run['Hidden Unit Multiplier'])
    
    model_d.compile(loss='binary_crossentropy', optimizer=current_run['Optimizer'], metrics=['acc'])

    batch_size_prop = current_run['Batch Size']

    timer_start = time.clock()
    history = model_d.fit(df_3_tr, y_train, validation_data=(df_3_val, y_valid),
                          verbose=0, epochs=current_run['Epochs'],
                          batch_size=np.round(int(batch_size_prop * df_3_tr.shape[0])))
    time_elapsed = time.clock()-timer_start
    time_elapsed_norm = (time.clock()-timer_start)/df_3_tr.shape[0]
    model = model_d

    y_val_pred = np.round(model.predict(df_3_val))
    y_tr_pred = np.round(model.predict(df_3_tr))
    timer_start = time.clock()
    y_ts_pred = np.round(model.predict(df_3_ts))
    time_elapsed_test = time.clock() - timer_start
    time_elapsed_test_norm = time_elapsed_test/df_3_ts.shape[0]
    
    #%% DNN aggregated to per-operational sample level
    #  Pool data from per-second level to obtain sample level insights
    tr_sample_labels = []
    tr_sample_labels_pred = []
    for f in np.unique(x_train['Sample']):
        tr_sample_labels.append(y_train[x_train['Sample'] == f].mode())
        tr_sample_labels_pred.append(np.round(y_tr_pred[x_train['Sample'] == f].mean()))
    tr_sample_labels = np.asarray(tr_sample_labels)
    tr_sample_labels_pred = np.asarray(tr_sample_labels_pred)

    v_sample_labels = []
    v_sample_labels_pred = []
    for f in np.unique(x_valid['Sample']):
        v_sample_labels.append(y_valid[x_valid['Sample'] == f].mode())
        v_sample_labels_pred.append(np.round(y_val_pred[x_valid['Sample'] == f].mean()))
    v_sample_labels = np.asarray(v_sample_labels)
    v_sample_labels_pred = np.asarray(v_sample_labels_pred)

    ts_sample_labels = []
    ts_sample_labels_pred = []
    for f in np.unique(df_ts['Sample']):
        ts_sample_labels.append(y_test[df_ts['Sample'] == f].mode())
        ts_sample_labels_pred.append(np.round(y_ts_pred[df_ts['Sample'] == f].mean()))
    ts_sample_labels = np.asarray(ts_sample_labels)
    ts_sample_labels_pred = np.asarray(ts_sample_labels_pred)

    print('Balanced Accuracy (Train):\t', balanced_accuracy_score(y_train, y_tr_pred))
    print('Balanced Accuracy (Val):\t', balanced_accuracy_score(y_valid, y_val_pred))
    print('Balanced Accuracy (Test):\t', balanced_accuracy_score(y_test, y_ts_pred))

    print('Balanced Accuracy Sample-level (Train):\t',
          balanced_accuracy_score(tr_sample_labels, tr_sample_labels_pred))
    print('Balanced Accuracy Sample-level (Val):\t', balanced_accuracy_score(v_sample_labels, v_sample_labels_pred))
    print('Balanced Accuracy Sample-level (Test):\t',
          balanced_accuracy_score(ts_sample_labels, ts_sample_labels_pred))

    conf_tr = confusion_matrix(y_train, y_tr_pred)
    conf_val = confusion_matrix(y_valid, y_val_pred)
    conf_ts = confusion_matrix(y_test, y_ts_pred)

    

    #%% Autoencoder 
    model_ae = Functions_Per_Second_Approach.sparse_ae(df_3_tr, current_run['Hidden Layers'],
                                                 current_run['Hidden Unit Multiplier'])
    model_ae.compile(loss='mse', optimizer=current_run['Optimizer'], metrics=['mse'])

    history = model_ae.fit(df_3_tr[x_train['Tool'] == 'Good'], df_3_tr[x_train['Tool'] == 'Good'],
                          validation_data=(df_3_val[x_valid['Tool'] == 'Good'], df_3_val[x_valid['Tool'] == 'Good']),
                          verbose=0, epochs=1000,
                          batch_size=np.round(int(batch_size_prop * df_3_tr.shape[0])))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    
    model = model_ae

    tr_pred = model.predict(df_3_tr)
    val_pred = model.predict(df_3_val)
    ts_pred = model.predict(df_3_ts)

    rms_tr = Functions_Per_Second_Approach.rms(tr_pred-df_3_tr.values)
    rms_val = Functions_Per_Second_Approach.rms(val_pred-df_3_val.values)
    rms_ts = Functions_Per_Second_Approach.rms(ts_pred-df_3_ts.values)

    residual_tr = (np.abs(tr_pred - df_3_tr)).mean(axis=1)
    residual_val = (np.abs(val_pred - df_3_val)).mean(axis=1)
    residual_test = (np.abs(ts_pred - df_3_ts)).mean(axis=1)

    thr = np.quantile(residual_tr,0.3) # Base the threshold on the approximate number of positive vs negative samples in data
    
    y_tr_pred_ae = np.ones(shape=y_train.shape)
    y_tr_pred_ae[residual_tr > thr] = 0
    y_val_pred_ae = np.ones(shape=y_valid.shape)
    y_val_pred_ae[residual_val > thr] = 0
    y_ts_pred_ae = np.ones(shape=y_test.shape)
    y_ts_pred_ae[residual_test > thr] = 0

    print('Balanced Accuracy AE (Train):\t', balanced_accuracy_score(y_train, y_tr_pred_ae))
    print('Balanced Accuracy AE (Valid):\t', balanced_accuracy_score(y_valid, y_val_pred_ae))
    print('Balanced Accuracy AE (Test):\t', balanced_accuracy_score(y_test, y_ts_pred_ae))
    
    conf_tr_ae = confusion_matrix(y_train, y_tr_pred_ae)
    conf_val_ae = confusion_matrix(y_valid, y_val_pred_ae)
    conf_ts_ae = confusion_matrix(y_test, y_ts_pred_ae)
    
    #%% Aggregation of all results for saving
    vals = pd.DataFrame(data=[[balanced_accuracy_score(y_train, y_tr_pred),
                               balanced_accuracy_score(y_valid, y_val_pred),
                               balanced_accuracy_score(y_test, y_ts_pred),
                               balanced_accuracy_score(tr_sample_labels, tr_sample_labels_pred),
                               balanced_accuracy_score(v_sample_labels, v_sample_labels_pred),
                               balanced_accuracy_score(ts_sample_labels, ts_sample_labels_pred),
                               conf_tr, conf_val, conf_ts, time_elapsed, time_elapsed_norm,
                               time_elapsed_test, time_elapsed_test_norm,
                               balanced_accuracy_score(y_train, y_tr_pred_ae),
                               balanced_accuracy_score(y_valid, y_val_pred_ae),
                               balanced_accuracy_score(y_test, y_ts_pred_ae),
                               conf_tr_ae, conf_val_ae, conf_ts_ae]], columns=outs.columns)
        
    outs = pd.concat([outs, vals], axis=0, ignore_index=True)
    
    if single_case:
        f_imp = Functions_Per_Second_Approach.feature_importance(model, df_3_tr, y_tr_pred, y_train, name='Case_'+str(i))
        Functions_Per_Second_Approach.plot_deep_model(history, plot=False)
        f_train = Functions_Per_Second_Approach.plot_confusion_matrix(y_train, y_tr_pred, classes=np.array(['Bad', 'Good']),
                                                      title='Training')
        f_val = Functions_Per_Second_Approach.plot_confusion_matrix(y_valid, y_val_pred, classes=np.array(['Bad', 'Good']),
                                                    title='Validation')
        f_test = Functions_Per_Second_Approach.plot_confusion_matrix(y_test, y_ts_pred, classes=np.array(['Bad', 'Good']),
                                                     title='Testing')
        f_train.figure.savefig('Training', bbox_inches='tight', dpi=300)
        f_test.figure.savefig('Testing', bbox_inches='tight', dpi=300)

master_out = pd.DataFrame(columns=[DOE.columns.append(outs.columns)])
if single_case:
    print('Results not saved')
else:
    master_out = pd.concat([DOE, outs], axis=1)
master_out.to_csv('DOE Results ' + str(datetime.now().date()) + '_1' + '.csv')
