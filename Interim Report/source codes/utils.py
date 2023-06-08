#!/usr/bin/env python
# coding: utf-8

# # Helper Functions

# In[ ]:


import pandas as pd
from math import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from random import sample
import os


# In[ ]:


def shuffling(data): #works
    shuf_data = data.sample(frac = 1).reset_index()
    shuf_data.pop('index')
    return shuf_data


# In[ ]:


def standardization(data, label): #works, very very slow
    norm_data = data.copy()
    row_num = norm_data.shape[0] 
    for column in norm_data.columns:
        col_obj = norm_data[column]
        mu_m = col_obj.sum()/row_num
        std_dev = col_obj.std()
        col_obj = (col_obj - mu_m)/std_dev
        norm_data[column] = col_obj
    norm_data["y"] = label
    return data


# In[ ]:


def z_norm(data, label): #works, very slow
    data_z_norm = data.copy()
    for column in data_z_norm.columns: #or don't put .columns, doesn't matter but it becomes slower
        data_z_norm[column] = (data_z_norm[column] - data_z_norm[column].mean())/data_z_norm[column].std()
    data_z_norm["y"] = label
    return data_z_norm 


# In[ ]:


def normalization(data, label): #works, very fast
    means = data.mean()
    std_devs = data.std()
    nor_data = (data - means)/std_devs
    nor_data["y"] = label
    return nor_data


# In[ ]:


def traintestsplit(data,ratio): #works
    row_num = data.shape[0] #row number
    label = data["y"].to_numpy()
    data.pop("y")
    data_n = data.to_numpy()
    if row_num%10 == 0 or row_num%100 == 0 or row_num%1000 == 0:
        test_num = int(row_num*ratio)
        X_test = data_n[0:test_num,:]
        y_test = label[0:test_num]
        X_train = data_n[test_num:,:]
        y_train = label[test_num:]
        return X_train, X_test, y_train, y_test
    else:
        test_num = int(floor(row_num*ratio))
        X_test = data_n[0:test_num,:]
        y_test = label[0:test_num]
        X_train = data_n[test_num:,:]
        y_train = label[test_num:]
        return X_train, X_test, y_train, y_test


# In[ ]:


def feature_extract(data):
    max_data = data.max(axis=1)
    max_data_s= max_data**2
    max_data_c= max_data**3
    
    min_data = data.min(axis=1)
    min_data_s= min_data**2
    min_data_c= max_data**3
    
    max_min_dif= max_data-min_data
    max_min_dif_s= max_min_dif**2
    
    data["MAX"] = max_data
    data["MIN"] = min_data
    data["MAX2"]= max_data_s
    data["MIN2"]= min_data_s
    data["MAX3"]= max_data_c
    data["MIN3"]= min_data_c
    data["MxMnDif"] =  max_min_dif
    data["MxMnDif2"] =  max_min_dif_s
    
    return data
    

