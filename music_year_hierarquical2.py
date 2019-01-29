'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

from compute_cost import compute_cost
from gradient_descent import plot_history, plot_history_train_validation
from music_data import MusicData
from normal_equation import solve_normal_equation
from normalize import z_norm
import numpy as np
from compute_accuracy import compute_accuracy, compute_accuracy_year


if __name__ == '__main__':
    '''
    Read music data.
    '''
    
    print "Loading data."
    #music_train = MusicData("resources/YearPredictionMSD_samples_train.txt")
    #music_test = MusicData("resources/YearPredictionMSD_samples_test.txt")
    music_train = MusicData("resources/YearPredictionMSD_train.txt")
    #music_train.add_features(2)
    #music_test.add_features(max_degree)
    #music_validation.add_features(max_degree)
    
    # Normalize data.    
    print "Normalize data."
    #music_train.X = z_norm(music_train.X)
  
    #music_validation = MusicData("resources/YearPredictionMSD_validation.txt")
    #music_test = MusicData("resources/YearPredictionMSD_test.txt")
    
    year1 = 1985
    year2 = 2000
    
    for year1 in range(1980, 2010):
        for year2 in range(2000, 2010):
            print year1, year2
            delta_year = 5
            less_year = music_train.y <= year1
            less_year.shape = (len(music_train.y))
            greater_year = music_train.y > year1
            greater_year.shape = (len(music_train.y))
            
            music_train_y_year_yes_or_not = np.array(music_train.y)
            music_train_y_year_yes_or_not[less_year] = year1
            music_train_y_year_yes_or_not[greater_year] = year2
            music_train_y_year_yes_or_not.shape = (len(music_train_y_year_yes_or_not), 1)
            
            # < year or > year classifier.    
            theta_year_yes_or_not = solve_normal_equation(music_train.X, music_train_y_year_yes_or_not, 0)
            
            # < year classifier.
            y = np.array(music_train.y[less_year])
            y.shape = (len(y), 1)
            X = music_train.X[np.where(less_year)]
            theta_year_less = solve_normal_equation(X, y, 0)
            print compute_accuracy(X, y, theta_year_less, delta_year)
            
            # > year classifier.    
            y = np.array(music_train.y[greater_year])
            y.shape = (len(y), 1)
            X = music_train.X[np.where(greater_year)]
            theta_year_more = solve_normal_equation(music_train.X[greater_year], y, 0)
            print compute_accuracy(X, y, theta_year_more, delta_year)
            
            print compute_accuracy_year(music_train.X, music_train.y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, year2, delta_year)
            #print compute_accuracy_year(music_validation.X, music_validation.y, theta_year_yes_or_not, theta_year_less, theta_year_more, year1, year2, delta_year)
            