'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

import copy

from compute_accuracy import compute_accuracy, compute_accuracy_year, \
    compute_accuracy_year_2, compute_cost_year_2
from compute_cost import compute_cost
from gradient_descent import plot_history, plot_history_train_validation
from music_data import MusicData
from normal_equation import solve_normal_equation
from normalize import z_norm, z_norm_by_feature
import numpy as np


if __name__ == '__main__':
    '''
    Read music data.
    '''
    
    print "Loading data."
    music_train = MusicData("resources/YearPredictionMSD_train.txt")
    music_validation = MusicData("resources/YearPredictionMSD_validation.txt")
    music_test = MusicData("resources/YearPredictionMSD_test.txt")

    
    music_train.add_features(3)
    music_test.add_features(3)
    music_validation.add_features(3)
    
    # Normalize data.
    print "Normalize data."
    # music_train.X = z_norm(music_train.X)
    # music_validation.X = z_norm(music_validation.X)
    # music_test.X = z_norm(music_test.X)
    music_train.X, mean_X, std_X = z_norm_by_feature(music_train.X)
    music_validation.X = z_norm_by_feature(music_validation.X, mean_X, std_X)
    music_test.X = z_norm_by_feature(music_test.X, mean_X, std_X)
    

    
    #music_validation = MusicData("resources/YearPredictionMSD_validation.txt")
    #music_test = MusicData("resources/YearPredictionMSD_test.txt")
    
    copy_music_train = copy.deepcopy(music_train) 
    #copy_music_train.balance_data_undersampling_custom(500)
    
    for year1 in range(1980, 2000):
        tirar = 20
        delta_year = 9
        less_year = music_train.y <= year1
        less_year.shape = (len(music_train.y))
        greater_year = music_train.y > year1
        greater_year.shape = (len(music_train.y))
            
        # < year or > year classifier.
        music_train_y_year_yes_or_not = np.array(copy_music_train.y)
        music_train_y_year_yes_or_not.shape = (len(music_train_y_year_yes_or_not), 1)
        copy_music_train.y = music_train_y_year_yes_or_not
                
        theta_year_yes_or_not = solve_normal_equation(copy_music_train.X, copy_music_train.y, 0)
        
        # < year classifier.
        y = np.array(music_train.y[less_year])
        y.shape = (len(y), 1)
        X = music_train.X[np.where(less_year)]
        theta_year_less = solve_normal_equation(X, y, 0)
        #print compute_accuracy(X, y, theta_year_less, delta_year)
        
        # > year classifier.    
        y = np.array(music_train.y[greater_year])
        y.shape = (len(y), 1)
        X = music_train.X[np.where(greater_year)]
        theta_year_more = solve_normal_equation(music_train.X[greater_year], y, 0)
        #print compute_accuracy(X, y, theta_year_more, delta_year)
        
        for delta_year in range(0, 10):
            print delta_year
    
            print compute_accuracy_year_2(music_train.X, music_train.y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, delta_year)
            print compute_accuracy_year_2(music_test.X, music_test.y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, delta_year)
            print compute_accuracy_year_2(music_validation.X, music_validation.y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, delta_year)
        
        print year1,compute_cost_year_2(music_test.X, music_test.y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, delta_year)
        