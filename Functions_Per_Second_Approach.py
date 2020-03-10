# -*- coding: utf-8 -*-
"""
Required Functions for the per second approach
"""
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels
from keras.metrics import binary_crossentropy
import seaborn as sns

def rms(x):
    x=np.array(x)
    return np.sqrt(np.vdot(x[np.where(~np.isnan(x))], x[np.where(~np.isnan(x))])/x[np.where(~np.isnan(x))].size)


def feature_importance(model, x, y_model, y_tr, name='Test'):
    err_original = mean_squared_error(y_model, y_tr.values)
    n_times = 100
    f_i = np.zeros([x.shape[1], n_times])
    for j in range(n_times):
        for i in range(x.shape[1]):
            temp = x.copy()
            temp.iloc[:, i] = np.random.permutation(x.iloc[:, i].values)
            y_temp = np.round(model.predict(temp))
            err = mean_squared_error(y_temp, y_tr.values)
            f_i[i, j] = np.abs(err - err_original)
    f_new = np.mean(f_i, axis=1)
    f_new2 = normalize(f_new.reshape(f_new.shape[0], 1), axis=0)

    plt.title('Relative Importance of Features')
    plt.xlabel('Relative Importance')
    sns.barplot(y=x.columns, x=f_new2.flatten(), palette='terrain')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    return f_new


def deep_fully_connected(x, n_layers=2, u_mult=2):

    n_inp = x.shape[1]
    inp = Input(shape=(n_inp,))
    act = 'relu'
    layer = BatchNormalization()(inp)
    for i in range(n_layers):

        layer = Dense(int(round(n_inp*u_mult)), activation=act)(layer)
        layer = BatchNormalization()(layer)

    layer = Dropout(0.9)(layer)
    layer = Dense(2, activation=act)(layer)
    final = Dense(1, activation='sigmoid')(layer)

    # this model maps an input to its output
    model = Model(inp, final)

    return model

def plot_deep_model(history, plot=True):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train', 'Test'], loc='upper left')

    if plot:
        plt.show()
    else:
        name='Model Accuracy'
        plt.title(name)
        plt.savefig(name,bbox_inches='tight',dpi=300)
        plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train', 'Test'], loc='upper left')
    if plot:
        plt.show()
    else:
        name='Model Loss'
        plt.title(name)
        plt.savefig(name,bbox_inches='tight',dpi=300)
        plt.close()

    return 0


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, plot=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.close()

    return ax

def sparse_ae(x, n_layers=2, u_mult=2):
    
    n_inp = x.shape[1]
    inp = Input(shape=(n_inp,))
    act = 'relu'
    layer = BatchNormalization()(inp)
    for i in range(n_layers):

        layer = Dense(int(round(n_inp*u_mult)), activation=act)(layer)
        layer = BatchNormalization()(layer)

    final = Dense(n_inp, activation='linear')(layer)
    model = Model(inp, final)
    return model
#    act = 'elu'
#    inp = Input(shape=(n_inp,))
#
#    layer = Dense(round(n_inp*1.5), activation=act)(inp)
#    layer = Dense(round(n_inp*2), activation=act)(layer)
#    layer = Dense(round(n_inp*2.5), activation=act)(layer)
#    layer = Dense(round(n_inp*5), activation=act)(layer)
#    layer = Dense(round(n_inp*15), activation=act)(layer)
#    layer = Dense(round(n_inp*5), activation=act)(layer)
#    layer = Dense(round(n_inp*2.5), activation=act)(layer)
#    layer = Dense(round(n_inp*2), activation=act)(layer)
#    layer = Dense(round(n_inp*1.5), activation=act)(layer)
#    decoded = Dense(n_inp, activation='linear')(layer)

    # this model maps an input to its reconstruction