"""
Created on Sat Jul 13 08:43:52 2019

@author: agharbi

The structure of the main code is done following the example https://github.com/hfawaz/dl-4-tsc
"""
import numpy as np
import os
import sys

def fit_classifier():
    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))
    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)
    classifier.fit(x_train,y_train,x_test,y_test)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = True):
    if classifier_name=='cnn':
        from models import cnn
        return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mlp':
        from  models import  mlp
        return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='rnn':
        from models import rnn
        return rnn.Classifier_RNN(output_directory,verbose)

# change this directory for your machine
root_dir = '/Users/agharbi/Desktop/Genos Study/'
data_dir = '/Users/agharbi/Desktop/Genos Study/Data'

# this is the code used to launch an experiment on a dataset
dataset_name = sys.argv[1]
classifier_name=sys.argv[2]


directory_path = root_dir+'/results/'+classifier_name+'/'+ dataset_name+'/'

output_directory = os.makedirs(directory_path)

if output_directory is None:
    print('Already done')
else:
    
    dir_name = '/Users/agharbi/Desktop/Genos Study/Data/' + dataset_name + '/'
    
    x_train = np.load(dir_name + 'x_train.npy')
    y_train = np.load(dir_name + 'y_train.npy')
    x_test = np.load(dir_name + 'x_test.npy')
    y_test = np.load(dir_name + 'y_test.npy')
    
    fit_classifier()

