# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:24:50 2020

@author: klaus
"""
#import of all necessary packages
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

#function to set the start working directory manually
#not necessary, if the .py file is executed in the directory of the .csv files
#os.chdir('D:\\OneDrive - bwedu\\Uni\\09 ABC 1\\Neuronale Netzwerke\\Python\\Wein')

#load raw data from the .csv files as panda dataframe
RW_orig = pd.read_csv('Rotwein.csv')
WW_orig = pd.read_csv('Weisswein.csv')

#convert panda dataframe to a numpy array
RW = pd.DataFrame.to_numpy(RW_orig)
WW = pd.DataFrame.to_numpy(WW_orig)

#create a new variabele for the two categories red (0) and white wine (1)
RW_num =  np.zeros((RW.shape[0],1), dtype = int) 
WW_num = np.ones((WW.shape[0],1), dtype = int)

#add the new variable two the datasets
RW = np.hstack((RW_num, RW))
WW = np.hstack((WW_num, WW))

#Merge the two datasets
W = np.vstack((RW, WW))

#shuffle the dataset
W = shuffle(W)

#seperate the sample columns from the column with the labels (winequality)
W_samples = W[:,0:12]
W_labels = W[:,12]  

#scale the sample variables in a range from 0 to 1 (ANN can only work with numbers from 0 to 1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_W_samples = scaler.fit_transform(W_samples)

#sort the wines into different categories, depending on the wine quality
#first category: all wines with a quality of 5 or below
#second category: all wines with a quality of 6
#third category: all wines with a quality of 7 or higher
#this is only necessary for some simpler experiments
cat_W_labels = np.array(range(0,(W_labels.shape[0])))
for i in range(0, (W_labels.shape[0])):
	if(W_labels[i] <= 5):
		cat_W_labels[i] = 0
	elif(W_labels[i] <= 6):
		cat_W_labels[i] = 1
	else:
		cat_W_labels[i] = 2
	pass


#seperate the dataset into a training and a test dataset	
W_train_samples = scaled_W_samples[0:5800,]
W_test_samples = scaled_W_samples[5800:6497,]
W_train_labels_cat = cat_W_labels[0:5800,]
W_test_labels_cat = cat_W_labels[5800:6497,]
W_train_labels_cont = W_labels[0:5800,]
W_test_labels_cont = W_labels[5800:6497,]

#save the current working directory
current_wd = os.getcwd()
#check if a directory 'prep_dataset' excist in the current working directory, if not then create it
if os.path.isdir('prep_dataset') is False:
    os.makedirs('prep_dataset')
#change the working directory to the directory 'prep_dataset'
os.chdir('prep_dataset')

#export all the numpy arrays as .csv file, by converting it to a panda dataframe and then exporting as a .csv file
pd.DataFrame(W_train_samples).to_csv("wine_train_samples.csv")
pd.DataFrame(W_test_samples).to_csv("wine_test_samples.csv")
pd.DataFrame(W_train_labels_cat).to_csv("wine_train_labels_cat.csv")
pd.DataFrame(W_test_labels_cat).to_csv("wine_test_labels_cat.csv")
pd.DataFrame(W_train_labels_cont).to_csv("wine_train_labels_cont.csv")
pd.DataFrame(W_test_labels_cont).to_csv("wine_test_labels_cont.csv")

#change the working directory back to the original working directory
os.chdir(current_wd)