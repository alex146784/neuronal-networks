# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:14:51 2020

@author: klaus
"""

#import all necessary packages
import os
import csv
import numpy as np
import pandas as pd
#from numpy import genfromtxt
#from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

#%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import IPython
import kerastuner as kt

#function to set the start working directory manually
#not necessary, if the .py file is executed in the directory of the .csv files
#os.chdir('D:\\OneDrive - bwedu\\Uni\\09 ABC 1\\Neuronale Netzwerke\\Python\\Wein')

#activation of the GPU, for this model not necessary
'''physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)'''


#save the current working directory	 
current_wd = os.getcwd()
#change the working directory to the 'prep_dataset' directroy, where the prepared data a saved as .csv files
os.chdir('prep_dataset')
#import the prepared data as panda dataframes
W_train_samples = pd.read_csv('wine_train_samples.csv')
W_train_labels = pd.read_csv('wine_train_labels_cont.csv')

#convert panda dataframe to a numpy array
#leave out the first column, because this are the rownumernumbers
W_train_samples = pd.DataFrame.to_numpy(W_train_samples)[:,1:13]
W_train_labels = (pd.DataFrame.to_numpy(W_train_labels)[:,1]-3)/6 #scale the labels in a range form 0 to 1

#change the working directory back to the original working directory
os.chdir(current_wd)	 
	 
#the definition of the model building function
#here it is possible to specify the model, the most parameters
#with hp funcitions it is possible to specify a range of values, in which an optimizer can optimize the model
def model_builder(hp):
	hp_layers = hp.Int('layers', min_value = 1, max_value = 5)#the number of layers in a range from 1 to 5
	hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4, 1e-5]) #test different learning rates 
	
	#the input layer with 12 nodes
	inputs = keras.Input(shape=(12,))
	
	x = inputs
	for i in range(hp_layers):
		#for each layer a different number of nodes (16 to 160)
		x = keras.layers.Dense(units = hp.Int('units_' + str(i), min_value = 16, max_value = 160, step = 16), activation = hp.Choice('act_func_' + str(i), values = ["relu", "sigmoid"]))(x)
	
	#the output layer with 1 node (continous regression)
	outputs = keras.layers.Dense(1, activation = hp.Choice('act_func_output', values = ["relu", "sigmoid"]))(x)
	
	#set the model together
	model = keras.Model(inputs, outputs)
	
	#compile the model
	model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate), 
					loss = "mean_squared_error",
					metrics = ['mse'])
	
	return model #return the finished model



#set settings for the tuner
tuner = kt.Hyperband(model_builder,
                     objective = 'val_mse', 
                     max_epochs = 200,
                     factor = 5, #the higher, the faster the optimizing, but the smalles the probability to find the best model
							#!!!delete directory before optimizing again!!!
                     directory = 'tuner_wine_continuous_actfunc',
                     project_name = 'wine_quality_continuous')



#define the necessary callbacks 
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

#start the optimizer 
tuner.search(W_train_samples, W_train_labels, epochs = 200, batch_size = 6, validation_split = 0.1, callbacks = [early_stopping_cb])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

#print the results of the optimizer
print(f"""
		The hyperparameter search is complete.
		Learning rate: {best_hps.get('learning_rate')}
		Number of Layers: {best_hps.get('layers')}
		""")
for i in range(best_hps.get('layers')):
	print("Number of nodes in layer %i: %f" %(i, best_hps.get('units_' + str(i))))
	print("Activationfunction in layer %i:" %(i) + best_hps.get('act_func_' + str(i)) )
print("Activationfunction in outputlayer" + best_hps.get('act_func_output') )



#define the log-directory for the logfiles for tensorboard
root_logdir = os.path.join(os.curdir, "tensorboard_wine_quality_continuous")
def get_run_logdir():
	import time
	run_id = 'wine_quality_continuous_' + time.strftime("run_%Y_%m_%d-%H_%M_%S")
	return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'

#define the necessary callbacks
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("wine_quality_continious_actfunc.h5", save_best_only=True)
early_stopping_cb2 = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)

#train the model
history = model.fit(W_train_samples, W_train_labels, batch_size = 6, epochs = 1000, validation_split = 0.1, callbacks = [checkpoint_cb, early_stopping_cb2, tensorboard_cb])	

