'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''
import numpy as np
from predict import predict_all, predict


def compute_accuracy(X, y, theta, year = 0):
    '''
    Compute accuracy.
    '''
    
    h_theta = np.round(predict_all(X, theta))
    delta = np.abs(h_theta -y)
    accuracy = float(sum(np.less_equal(delta, year)) * 1.0 / len(h_theta))
    
    return accuracy    

def compute_accuracy_year(X, y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, year2, year_delta):
    '''
    Compute accuracy.
    '''
    
    accuracy = 0
    year_yes_or_not = np.round(predict_all(X, theta_year_yes_or_not))
    
    for index in range(len(year_yes_or_not)): 
        if abs(year_yes_or_not[index] -year2) < abs(year_yes_or_not[index] -year1):
            h_theta = np.round(predict(X[index], theta_year_more))
        else:
            h_theta = np.round(predict(X[index], theta_year_less))
        
        if abs(h_theta - y[index]) < year_delta:
            accuracy += 1.0
            
    return accuracy/len(X)

def compute_accuracy_year_2(X, y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, year_delta):
    '''
    Compute accuracy.
    '''
    
    accuracy = 0
    year_yes_or_not = np.round(predict_all(X, theta_year_yes_or_not))
    
    for index in range(len(year_yes_or_not)): 
        if year_yes_or_not[index] > year1:
            h_theta = np.round(predict(X[index], theta_year_more))
        else:
            h_theta = np.round(predict(X[index], theta_year_less))
        
        if np.round(abs(h_theta - y[index])) <= year_delta:
            accuracy += 1.0
            
    return accuracy/len(X)

def compute_cost_year_2(X, y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, year_delta):
    '''
    Compute accuracy.
    '''
    
    cost = 0
    year_yes_or_not = np.round(predict_all(X, theta_year_yes_or_not))
    
    for index in range(len(year_yes_or_not)): 
        if year_yes_or_not[index] > year1:
            h_theta = np.round(predict(X[index], theta_year_more))
        else:
            h_theta = np.round(predict(X[index], theta_year_less))
            
        delta = np.square(h_theta - y[index])
    
        cost += delta
        
    cost = cost / (2.0 * len(X))
    
    return cost