#!/usr/bin/env python
# coding: utf-8

# In[414]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


# In[474]:


def kfold(data, r):
    
    aux = data.copy()
    size = math.ceil(len(aux)/r)
        
    fold = {}
        
    for i in range(r):
        
        fold[i] = {}
            
        if(i == r - 1):
            
            fold[i] = data
                
        else:
            
            samp = data.sample(n = size)
            fold[i] = samp
            
            remov = pd.concat([data, samp]) 
            data = remov.drop_duplicates(keep=False)

    return fold

def cross_validation(data, r):
    
    data = data.drop_duplicates(keep='first')
    
    folds = kfold(data, r)
    partitions = {}
    folder = pd.DataFrame()
        
    for i in range(len(folds)):

        folder = pd.concat([folder, folds[i]])
            
    for i in range(len(folds)):
        
        test = folds[i]
        
        train = pd.concat([folder, folds[i]])
        train = train.drop_duplicates(keep=False)
        
        partitions[i] = []
        partitions[i].append(train) 
        partitions[i].append(test) 
    
    return partitions


# In[476]:


def leave_one_out(data):
    sample = {}
    sample[0] = []
    
    train, test = train_test_split(data, test_size = 1)
    
    sample[0].append(train)
    sample[0].append(test)
    
    return sample


# In[477]:


def random_subsampling(data, n):
    
    subs = {}
    
    for i in range(n):
    
        train, test = train_test_split(data, test_size = 0.25)
        
        subs[i] = [train, test]
    
    return subs

