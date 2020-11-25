# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:51 2020

@author: klaus
"""
import os
import csv
import numpy as np
import pandas as pd
#from numpy import genfromtxt
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
#%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

'''physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	 
	 
	 
 
RW_orig = pd.read_csv('Rotwein.csv')
WW_orig = pd.read_csv('Weisswein.csv')

RW = pd.DataFrame.to_numpy(RW_orig)
WW = pd.DataFrame.to_numpy(WW_orig)

RW_num =  np.zeros((RW.shape[0],1), dtype = int) 
WW_num = np.ones((WW.shape[0],1), dtype = int)

RW = np.hstack((RW_num, RW))
WW = np.hstack((WW_num, WW))

W = np.vstack((RW, WW))

W = shuffle(W)


W_samples = W[:,0:12]
W_labels = W[:,12]




scaler = MinMaxScaler(feature_range=(0,1))
scaled_W_samples = scaler.fit_transform(W_samples)
cat_W_labels = np.array(range(0,(W_labels.shape[0])))
for i in range(0, (W_labels.shape[0])):
	if(W_labels[i] <= 5):
		cat_W_labels[i] = 0
	elif(W_labels[i] <= 6):
		cat_W_labels[i] = 1
	else:
		cat_W_labels[i] = 2
	pass

	
W_train_samples = scaled_W_samples[0:5800,]
#RW_valid_samples = scaled_RW_samples[999:1299,]
W_test_samples = scaled_W_samples[5800:6497,]

W_train_labels = cat_W_labels[0:5800,]
#RW_valid_labels = cat_RW_labels[999:1299,]
W_test_labels = cat_W_labels[5800:6497,]

model = Sequential([
	Dense(units = 50, input_shape=(12,), activation='relu'),
	Dense(units = 50, activation = 'relu'),
	Dense(units = 50, activation = 'relu'),
	Dense(units = 3, activation='softmax')
])

model.compile(optimizer = Adam(learning_rate=0.0001),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)


#class_weight = {0: 1.19, 1: 1., 2: 2.23}

class_weight = {0: 1., 1: 1., 2: 1.5}

model.fit(x = W_train_samples,
			 y = W_train_labels,
			 validation_split = 0.1,
			 batch_size = 6,
			 epochs = 50,
			 shuffle = True,
			 verbose = 2,
			 class_weight = class_weight,
			 )



predictions = model.predict(x = W_test_samples, batch_size = 10, verbose = 1 )
rounded_predictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(y_true=W_test_labels, y_pred=rounded_predictions)

cm_plot_labels = ['bad wine','medium wine', 'good wine']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


#for i in range(1,11):
#	print(i,": ", np.count_nonzero(W_labels == i))