#!/usr/bin/env python
# coding: utf-8

# # Data Visualization

# In[97]:


import pandas as pd
from math import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from random import sample
import os
import random


# In[98]:


epilepsy_data = pd.read_csv("bonn_epilepsy.csv", sep =",")

epilepsy_data.drop("Unnamed",axis=1,inplace=True)
epilepsy_data.y = epilepsy_data.y==1
epilepsy_data.y = epilepsy_data.y.astype(int)

epilepsy_data= epilepsy_data[epilepsy_data.isnull().any(axis=1)==False]
label = epilepsy_data["y"].astype("category").to_numpy()
epileptic_data=epilepsy_data.drop("y",axis=1,inplace=False).to_numpy()


# In[99]:


patient_list=epilepsy_data.index[epilepsy_data["y"] == 1].tolist()
healthy_list = epilepsy_data.index[epilepsy_data["y"] == 0].tolist()
random.shuffle(patient_list)
random.shuffle(healthy_list)


# In[100]:


for p in range(2):
    plt.subplot(2, 2, 2*p+1)
    plt.plot(epileptic_data[patient_list[p],:])
    plt.title("Patient")
    plt.subplot(2, 2, 2*(p+1))
    plt.plot(epileptic_data[healthy_list[p],:])
    plt.title("Healthy")
    plt.subplots_adjust(bottom=0.5, right=2, top=2)


# In[ ]:




