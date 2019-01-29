'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

from compute_accuracy import compute_accuracy
from compute_cost import compute_cost
from gradient_descent import plot_history, plot_history_train_validation
from music_data import MusicData
from normal_equation import solve_normal_equation
from normalize import z_norm
import numpy as np


if __name__ == '__main__':
    '''
    Read music data.
    '''
    
    print "Loading data."
    #music_train = MusicData("resources/YearPredictionMSD_samples_train.txt")
    #music_test = MusicData("resources/YearPredictionMSD_samples_test.txt")
    music_train = MusicData("resources/YearPredictionMSD_train.txt")
    music_validation = MusicData("resources/YearPredictionMSD_validation.txt")
    music_test = MusicData("resources/YearPredictionMSD_test.txt")
    max_degree = 3
    
    # Add polynomial features.
    print "Adding polynomial features."
    music_train.add_features(max_degree)
    music_test.add_features(max_degree)
    music_validation.add_features(max_degree)
    
    ## Normalize data.    
    print "Normalize data."
    #music_train.X = z_norm(music_train.X)
    #music_validation.X = z_norm(music_validation.X)
    #music_test.X = z_norm(music_test.X)
    #music_train.X, mean_X, std_X = z_norm_by_feature(music_train.X)
    #music_validation.X = z_norm_by_feature(music_validation.X, mean_X, std_X)
    #music_test.X = z_norm_by_feature(music_test.X, mean_X, std_X)
   
    
    # Balacing train data.
    print "Balacing train data."
    before_balacing_size = len(music_train.X)
    #music_train.balance_data_oversampling_random()
    #music_train.balance_data_oversampling_smote_borderline1()
    #music_train.balance_data_oversampling_smote_borderline2()
    #music_train.balance_data_oversampling_smote_regular()
    #music_train.balance_data_oversampling_smote_svm()
    #music_train.balance_data_ensemblesampling_condensed_nearest_neighbour()
    #music_train.balance_data_undersampling_random()
    #music_train.balance_data_undersampling_cluster_centroids()
    #music_train.balance_data_undersampling_tomek_links()
    music_train.balance_data_ensemblesampling_balance_cascade()
    #music_train.balance_data_ensemblesampling_balance_cascade()
    after_balacing_size = len(music_train.X)
    print "Before balacing size: " + str(before_balacing_size)
    print "After balacing size: " + str(after_balacing_size)
    
    # Set train parameters.
    lambdav = 0.0000000001
    n = len(music_train.X[0])
    
    print "Solving normal equation."
    
    # Get thetas to reduce data.
    theta = solve_normal_equation(music_train.X, music_train.y, lambdav)
    ordered_theta = np.argsort(np.abs(theta).reshape(len(theta)))
    ordered_theta = ordered_theta[::-1]
    
    # Initialize costs.
    J_history_train = np.zeros(n)
    J_history_validation = np.zeros(n)
    
    for iteration in range(n):
        theta = solve_normal_equation(music_train.X[:, ordered_theta[:(n - iteration)]], music_train.y, lambdav)
        J_history_train[iteration] = compute_cost(music_train.X[:, ordered_theta[:(n - iteration)]], music_train.y, theta, 0)
        J_history_validation[iteration] = compute_cost(music_validation.X[:, ordered_theta[:(n - iteration)]], music_validation.y, theta, 0)
        
        print "Theta size: " + str(n - iteration)
        print "J_train: %f" % J_history_train[iteration]
        print "J_validation: %f" % J_history_validation[iteration]
        print "Accuracy: %f" % compute_accuracy(music_test.X[:, ordered_theta[:(n - iteration)]], music_test.y, theta, 9)

        ordered_theta = np.argsort(np.abs(theta).reshape(len(theta)))
        ordered_theta = ordered_theta[::-1]
    
    plot_history_train_validation(J_history_train, J_history_validation)
    plot_history(J_history_train -J_history_validation)
