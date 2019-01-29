'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np


def solve_normal_equation(X, y, lambdav):
    '''
    Perform gradient descent to learn theta.
    
    theta = (X'*X)^-1 * (X'*Y)
    '''
    
    X_transpose = np.transpose(X)
    inverse_squared_x = np.linalg.pinv(np.dot(X_transpose, X))
    n = len(inverse_squared_x)
    regularization_term = lambdav * np.eye(n)
    regularization_term[0][0] = 0
    theta = np.dot(inverse_squared_x + regularization_term, np.dot(X_transpose, y))  
    
    return theta
