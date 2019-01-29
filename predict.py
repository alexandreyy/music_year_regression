'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np

def predict(x, theta):
    '''
    Predict result
    '''
    
    return np.dot(x, theta)

def predict_all(X, theta):
    '''
    Predict result
    '''
    
    return np.dot(X, theta)
