'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np
from predict import predict_all


def compute_cost(X, y, theta, lambdav=0):
    '''
    Compute cost.
    
    Cost = (sum((h(X) -y)^2) + lambda * sum(theta.^2))/ (2*m)
    '''

    m = len(X)
    h_theta = predict_all(X, theta)
    delta = np.square(h_theta - y)
    cost = (np.sum(delta) + lambdav * np.sum(np.square(theta))) / (2 * m)
    
    return cost
