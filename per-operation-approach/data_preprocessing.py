"""
Created on Sat Jul 13 08:43:52 2019

@author: agharbi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# output directory 
root_dir = '/Users/agharbi/Desktop/Genos Study/'


# import data, df1 and df2 will be used for training and testing, df3 will be used for validation
dataframe1 = pd.read_csv(r'/Users/agharbi/Desktop/Genos Study/Data/_data_20190502.csv') 
dataframe2 = pd.read_csv(r'/Users/agharbi/Desktop/Genos Study/Data/_data_20190506.csv')


# encode tool quality 
cut_cols = {"Tool": {"Good": 1, "Bad": 0}}

dataframe1.replace(cut_cols, inplace=True)    
dataframe2.replace(cut_cols, inplace=True)


# create the different sets for experiments 
'''
Index(['Unnamed: 0', 'Timestamp', 'Test or Experiment', 'Current Mean',
       'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS',
       'Current Crest Factor', 'Current Peak', 'Vibration Mean',
       'Vibration Std', 'Vibration Skewness', 'Vibration Kurtosis',
       'Vibration RMS', 'Vibration Crest Factor', 'Vibration Peak', 'Sample',
       'LS1cmd', 'LS1load', 'LS1speed', 'LX1load', 'LX1actw', 'LZ1actw',
       'LZ1load', 'x', 'y', 'z', 'Cutting Depth (mm)',
       'Finishing Feed Rate (mm/rev)', 'Surface Speed (M/min)', 'Material',
       'Operation', 'Tool'],
      dtype='object')
'''


### all data 
f1 = ['Timestamp', 'Current Mean', 'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS', 'Current Crest Factor', 'Current Peak', 'Vibration Mean', 'Vibration Std', 'Vibration Skewness', 'Vibration Kurtosis', 'Vibration RMS', 'Vibration Crest Factor', 'Vibration Peak', 'Sample', 'LS1cmd', 'LS1load', 'LS1speed', 'LX1load', 'LZ1load', 'x', 'z', 'Cutting Depth (mm)', 'Finishing Feed Rate (mm/rev)', 'Surface Speed (M/min)', 'Tool']

### genos only data
f2 = ['Timestamp','Sample', 'LS1cmd','LS1load', 'LX1actw', 'LZ1actw','LS1speed', 'LX1load', 'LZ1load', 'x', 'z', 'Tool']

### genos + doe
f3 = ['Timestamp','Sample', 'LS1cmd', 'LS1load', 'LX1actw', 'LZ1actw','LS1speed', 'LX1load', 'LZ1load', 'x', 'z', 'Cutting Depth (mm)', 'Finishing Feed Rate (mm/rev)', 'Surface Speed (M/min)', 'Tool']

### sensor only data

f4 = ['Timestamp', 'Current Mean', 'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS', 'Current Crest Factor', 'Current Peak', 'Vibration Mean', 'Vibration Std', 'Vibration Skewness', 'Vibration Kurtosis', 'Vibration RMS', 'Vibration Crest Factor', 'Vibration Peak', 'Sample', 'Tool']

### sensor + doe
f5 = ['Timestamp', 'Current Mean', 'Current Std', 'Current Skewness', 'Current Kurtosis', 'Current RMS', 'Current Crest Factor', 'Current Peak', 'Vibration Mean', 'Vibration Std', 'Vibration Skewness', 'Vibration Kurtosis','Vibration RMS', 'Vibration Crest Factor', 'Vibration Peak', 'Sample', 'Cutting Depth (mm)', 'Finishing Feed Rate (mm/rev)', 'Surface Speed (M/min)', 'Tool']

ssc = StandardScaler()

# max length of a sequence
max_length = 112

feature_sets = [f1, f2, f3, f4, f5]
directories = ['all/', 'controller/', 'controller+doe/', 'sensors/', 'sensors+doe/']

for features, directory in zip(feature_sets, directories):    
    df1 = dataframe1[features]
    df2 = dataframe2[features]
    
    l1 = len(df1['Sample'].value_counts(sort=True))
    l2 = len(df2['Sample'].value_counts(sort=True))
    
    y_test = df1.groupby(['Sample', 'Tool']).median().index.get_level_values('Tool').values
    y_train = df2.groupby(['Sample', 'Tool']).median().index.get_level_values('Tool').values
    
    n_var = len(features)-2
    
    x_train = np.empty((l2, max_length, n_var))
    x_test = np.empty((l1,  max_length, n_var))
    
    train_samples = []
      
    j = 0
    
    for i in range(1, 145):
        im2 = df2.loc[df2['Sample'] == i]
        im2 = im2.drop(['Sample'], axis=1)
        im2 = im2.drop(['Tool'], axis=1)
        
        if im2.empty:
            continue   
        
        train_samples.append(i)
    
        # normalize all data
        im2 = ssc.fit_transform(im2)
        ls = im2.shape[0]
        if  ls < max_length:
            im2 = np.pad(im2, ((0, max_length-ls), (0, 0)), 'constant')
        x_train[j,] = im2
        j+=1
        
        
    j = 0 
    test_samples = []
    
    for i in range(1, 145):
        im1 = df1.loc[df1['Sample'] == i]
        im1 = im1.drop(['Sample'], axis=1)
        im1 = im1.drop(['Tool'], axis=1)
        if im1.empty:
            continue
        test_samples.append(i)
        # normalize all data
        im1 = ssc.fit_transform(im1)
        ls = im1.shape[0]
        if  ls < max_length:
            im1 = np.pad(im1, ((0, max_length-ls), (0, 0)), 'constant')
        
        x_test[j,] = im1
        j+=1
        
    out_dir = root_dir + directory

    # save them
    np.save(out_dir+'x_train.npy',x_train)
    np.save(out_dir+'y_train.npy',y_train)
    np.save(out_dir+'x_test.npy',x_test)
    np.save(out_dir+'y_test.npy',y_test)

