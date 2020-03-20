import numpy as np
import sys
import sklearn

def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)

    classifier.fit(x_train,y_train,x_test,y_test, y_true)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = True):
    if classifier_name=='fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mlp':
        from  classifiers import  mlp
        return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory,verbose)
    if classifier_name=='cnn': # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)

############################################### main

# change this directory for your machine
root_dir = '/Users/agharbi/Desktop/Processed Data Genos/'


# this is the code used to launch an experiment on a dataset
archive_name = sys.argv[1]
dataset_name = sys.argv[2]
classifier_name=sys.argv[3]


output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+\
    dataset_name+'/'

output_directory = create_directory(output_directory)

print('Method: ',archive_name, dataset_name, classifier_name, itr)

if output_directory is None:
    print('Already done')
else:

    datasets_dict = read_dataset(root_dir,archive_name,dataset_name)

    fit_classifier()

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory+'/DONE')