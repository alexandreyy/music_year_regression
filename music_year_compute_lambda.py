'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

from compute_cost import compute_cost
from gradient_descent import plot_history_train_validation
from music_data import MusicData
from normal_equation import solve_normal_equation
from normalize import z_norm_by_feature
import numpy as np


if __name__ == '__main__':
    '''
    Read music data.
    '''

    print "Loading data."
    # music_train = MusicData("resources/YearPredictionMSD_samples_train.txt")
    # music_test = MusicData("resources/YearPredictionMSD_samples_test.txt")
    music_train = MusicData("resources/YearPredictionMSD_train.txt")
    music_validation = MusicData("resources/YearPredictionMSD_validation.txt")
    music_test = MusicData("resources/YearPredictionMSD_test.txt")

    # Normalize data.
    print "Normalize data."
    # music_train.X = z_norm(music_train.X)
    # music_validation.X = z_norm(music_validation.X)
    # music_test.X = z_norm(music_test.X)
    music_train.X, mean_X, std_X = z_norm_by_feature(music_train.X)
    music_validation.X = z_norm_by_feature(music_validation.X, mean_X, std_X)
    music_test.X = z_norm_by_feature(music_test.X, mean_X, std_X)

    # Set train parameters.
    lambdav = 10
    number_iterations = 10

    print "Solving normal equation."
    J_history_train = np.zeros(number_iterations)
    J_history_validation = np.zeros(number_iterations)

    for iteration in range(number_iterations):
        lambdav /= 10.0
        theta = solve_normal_equation(music_train.X, music_train.y, lambdav)

        J_history_train[iteration] = compute_cost(music_train.X, music_train.y, theta, 0)
        J_history_validation[iteration] = compute_cost(music_validation.X, music_validation.y, theta, 0)

        # print J_history_train[iteration]
        print J_history_validation[iteration]
        # print compute_cost(music_test.X, music_test.y, theta, lambdav)

    plot_history_train_validation(J_history_train, J_history_validation)
