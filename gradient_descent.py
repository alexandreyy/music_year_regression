'''
Created on 19/08/2015

@author: Alexandre Yukio Yamashita
'''

from compute_cost import compute_cost
from normal_equation import solve_normal_equation
import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(X, y, theta, alpha, lambdav, number_iterations):
    '''
    Perform gradient descent to learn theta.
    
    h(X) = X*theta
    theta = theta -alpha*(X'*(h(X) -y) +lambda*theta) / m
    '''

    m = len(X)

    if theta == None:
        theta = np.zeros((len(X[0]), 1))

    for _iteration in range(number_iterations):
        d_J_d_Theta = np.dot(np.transpose(X), (np.dot(X, theta) - y))
        theta = theta - alpha * (d_J_d_Theta + lambdav * theta) / m

        if _iteration % 100 == 0:
            print compute_cost(X, y, theta)
        # print "[%d] %f" % (_iteration, compute_cost(X, y, theta))

    return theta

def gradient_descent_with_J_history(X, y, theta, alpha, lambdav, number_iterations):
    '''
    Perform gradient descent to learn theta.
    
    h(X) = X*theta
    theta = theta -alpha*X'*(h(X) -y) / m
    '''

    m = len(X)
    J_history = np.zeros(number_iterations)

    if theta == None:
        theta = np.zeros((len(X[0]), 1))

    for iteration in range(number_iterations):
        d_J_d_Theta = np.dot(np.transpose(X), (np.dot(X, theta) - y))
        theta = theta - alpha * (d_J_d_Theta + lambdav * theta) / m

        if iteration % 100 == 0:
            print compute_cost(X, y, theta)
        # print "[%d] %f" % (iteration, compute_cost(X, y, theta))

        J_history[iteration] = compute_cost(X, y, theta)

    return theta, J_history

def plot_data(X, y):
    '''
    Plot data.
    '''

    plt.plot(X, y, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_result(X, y, theta):
    '''
    Plot result.
    '''

    plt.plot(X[:, 1], y, 'ro')
    plt.plot(X[:, 1], np.dot(X, theta))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_history(J_history):
    '''
    Plot cost history.
    '''

    plt.plot(J_history)
    plt.xlabel('iteracao')
    plt.ylabel('J')
    plt.show()


def plot_history_train_validation(J_history_train, J_history_validation):
    '''
    Plot cost history.
    '''
    # plt.plot(J_history_train, 'r--', J_history_validation, 'b-')
    plt.plot(J_history_train, 'r-', J_history_validation, 'bs')
    plt.xlabel('lambda')
    plt.ylabel('J')
    plt.show()

if __name__ == '__main__':
    m = 100  # number of training examples.
    theta = np.array([10, 5])  # theta parameters.
    k = len(theta)  # number of parameters.
    theta.shape = (k, 1)

    # Generate random input features.
    X_min_value = 0
    X_max_value = 10
    X = np.hstack((np.ones((m, 1)), \
                   X_min_value + np.random.rand(m, 1) * (X_max_value - X_min_value)))

    # Generate output features.
    mean, sigma = 0, 0.15  # mean and standard deviation.
    error = np.random.normal(mean, sigma, m)
    error.shape = (m, 1)
    y = np.dot(X, theta) + error * 50

    # Perform gradient descent.
    found_theta_normal_equation = solve_normal_equation(X, y, 0)
    found_theta_gradient_descent, J_history = gradient_descent_with_J_history(X, y, None, 0.01, 0, 1500)

    print "Theta - Normal equation: "
    print found_theta_normal_equation

    print "Theta - Gradient descent: "
    print found_theta_gradient_descent

    # Plot cost history and result.
    plot_history(J_history)
    plot_result(X, y, found_theta_gradient_descent)
