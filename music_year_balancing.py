'''
Created on 30/08/2015

@author: Alexandre Yukio Yamashita
'''

from compute_accuracy import compute_accuracy
from compute_cost import compute_cost
from gradient_descent import gradient_descent, gradient_descent_with_J_history, \
    plot_history
from music_data import MusicData
from normal_equation import solve_normal_equation
from normalize import z_norm, z_norm_by_feature


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
    music_train.balance_data_oversampling_smote_regular()
    music_train.X, mean_X, std_X = z_norm_by_feature(music_train.X)
    #music_train.balance_data_undersampling_custom()
    music_validation.X = z_norm_by_feature(music_validation.X, mean_X, std_X)
    music_test.X = z_norm_by_feature(music_test.X, mean_X, std_X)

    # Balacing train data.
    # print "Balacing train data."
    # music_train.balance_data_oversampling_smote_regular()

    # Set train parameters.
    # lambdav = 0.00001
    lambdav = 0
    # alpha = 0.0000001
    # iterations = 1000000
    alpha = 0.1
    iterations = 1200

    # print "Solving normal equation."
    theta = solve_normal_equation(music_train.X, music_train.y, lambdav)

    print "Solving using gradient descent."
    # theta = gradient_descent(music_train.X, music_train.y, None, alpha, lambdav, iterations)
    #theta, J_history = gradient_descent_with_J_history(music_train.X, music_train.y, None, alpha, lambdav, iterations)
    #plot_history(J_history)

    print "Computing cost."
    print compute_cost(music_train.X, music_train.y, theta, lambdav)
    print compute_cost(music_validation.X, music_validation.y, theta, lambdav)
    print compute_cost(music_test.X, music_test.y, theta, lambdav)

    for delta_year in range(10):
        print delta_year

        print "Computing train accuracy."
        print compute_accuracy(music_train.X, music_train.y, theta, delta_year)
        print compute_accuracy(music_validation.X, music_validation.y, theta, delta_year)
        print compute_accuracy(music_test.X, music_test.y, theta, delta_year)
