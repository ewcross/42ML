# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_logistic_regression.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/09 12:12:13 by ecross            #+#    #+#              #
#    Updated: 2020/05/11 16:22:11 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from math import e as euler
from math import log
import matplotlib.pyplot as plt
import sys

class MyLogisticRegression:

    def __init__(self, thetas, alpha=0.001, n_cycle=1000, penalty='l2'):
        if not self.check_init(thetas, alpha, n_cycle):
            self.thetas = None
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = thetas
        if penalty == 'l2':
            self.gradient_ = self.gradient_l2_
            self.cost_ = self.cost_l2_
        elif penalty != None:
            print('invalid penalty paramter')
            raise ValueError
    
    def check_init(self, theta, alpha, n_cycle):
        if type(theta) != np.ndarray:
            print("theta must be 1D numpy array")
            return 0
        if type(alpha) != float and type(alpha) != int:
            print("alpha must be float or int")
            return 0
        if type(n_cycle) != int:
            print("n_cycle must be int")
            return 0
        return 1

    def fit_(self, x, y):
        
        """carries out logistic regression given a set of data features, xn,
        training output values, y, and initial theta values (self.thetas),
        returning a vector containing optimised theta values"""

        if type(x) != np.ndarray or type(y) != np.ndarray:
            print("x, y and theta must be numpy ndarrays")
            return None
        for i in range(self.n_cycle):
            nabla = self.gradient_(x, y)
            if nabla is None:
                print('fit: error calculating gradient vector')
                return None
            self.thetas = self.thetas - (nabla * self.alpha)
        return self.thetas

    def plot_convergence(self, x, y):

        """carries out logistic regression given a set of data features, xn,
        training output values, y, and initial theta values (self.thetas),
        returning a vector containing new theta values - this version plots
        the change in cost function with convergence cycles, and displays this"""

        if type(x) != np.ndarray or type(y) != np.ndarray:
            print("x, y and theta must be numpy ndarrays")
            return None
        j_values = []
        for i in range(self.n_cycle):
            nabla = self.gradient_(x, y)
            if nabla is None:
                print('error calculating gradient vector')
                return None
            self.thetas = self.thetas - (nabla * self.alpha)
            j_values.append(self.cost_(y, self.predict_(x)))
        j = np.array(j_values)
        epoc = np.array(np.arange(self.n_cycle))
        plt.plot(epoc, j, 'o')
        plt.show()
        return self.thetas

    def gradient_(self, x, y):

        """computes a 1d vector of the gradients of the cost function, the
        partial derivitive with respect to each theta, for given theta values"""
    
        #add column of 1s to right side of x value vector
        #number of theta values must match number of features (n in xn)
        #get 1d vector of y_hat (predicted) values, using the sigmoid function
        #check that y_hat (calculated from x) and y are compatible
        #dot product of transpose of x with (y_hat - y) vector gives vector of gradients
    
        y_hat = self.predict_(x)
        if y_hat is None:
            return None
        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            return None
        y_hat = y_hat - y
        return np.dot(self.add_intercept(x).transpose(), y_hat) * (1 / y.size)
        
    def gradient_l2_(self, x, y, lambda_=0.5):
        if type(lambda_) != float:
            print('lambda must be float')
            return None
        y_hat = self.predict_(x)
        if y_hat is None:
            return None
        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            return None
        y_hat = y_hat - y
        y_hat = np.dot(self.add_intercept(x).transpose(), y_hat) * (1 / y.size)
        th = np.array(self.thetas)
        th = th.reshape(-1,)
        th[0] = 0
        lambda_theta = th * lambda_
        return y_hat + (lambda_theta / y.shape[0])

    def predict_(self, x):

        """generates vector of predicted (y_hat) values, given an input
        vector of x values and a vector of thetas"""

        new = self.add_intercept(x)
        if self.thetas.ndim > 1:
            self.thetas = self.thetas.reshape((self.thetas.size,))
        if self.thetas.shape != (new.shape[1],):
            print("mismatch between theta vector size, and input features")
            return None
        return self.sigmoid_(np.dot(new, self.thetas))
    
    def sigmoid_(self, vec):

        """computes the sigmoid of a vector, returning a new vector of same shape"""
    
        if type(vec) != np.ndarray:
            print('sigmoid function needs a numpy array')
            return None
        if vec.size == 0:
            print('sigmoid function needs a non-empty numpy array')
            return None
        def sig(var):
            return 1 / (1 + euler**(-var))
        return np.vectorize(sig)(vec)
    
    def add_intercept(self, x):

        """takes a non-empty numpy.ndarray and returns a 2D matrix
        with a column of 1's concatenated to the right hand side
        of the  original vector"""

        if x.ndim == 1:
            x = x[:, None]
        return np.insert(x, 0, 1, axis=1)

    def cost_(self, y, y_hat, eps=1e-15):

        """calculates the cost of each predicted (y_hat) value in y_hat vector
        using the logistic loss function, with the actual values in the vector y"""

        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            print('cost function (log loss): y and y(predicted) do not match')
            return None
        vec_log = np.vectorize(log)
        ret = np.dot(y + eps, vec_log(y_hat + eps))
        ret += np.dot((1 - y) + eps, vec_log((1 - y_hat) + eps))
        return ret * (-1. / y.size)
    
    def cost_l2_(self, y, y_hat, lambda_=0.5, eps=1e-15):
        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            print('cost function (log loss): y and y(predicted) do not match')
            return None
        if type(lambda_) != float:
            print('lambda must be float')
            return None
        vec_log = np.vectorize(log)
        ret = np.dot(y + eps, vec_log(y_hat + eps))
        ret += np.dot((1 - y) + eps, vec_log((1 - y_hat) + eps))
        ret *= (-1. / y.size)
        theta = np.array(self.thetas)
        theta = theta.reshape(-1,)
        theta[0] = 0
        return (cost + ((lambda_ / y.shape[0]) * np.dot(theta, theta)))
    
    def check_size_and_shape(self, y, y_hat):
        if y.size == 0 or y_hat.size == 0:
            return 0, None, None
        if y.ndim > 1:
            y = y.reshape((y.size,))
        if y_hat.ndim > 1:
            y_hat = y_hat.reshape((y_hat.size,))
        if y.shape != y_hat.shape:
            return 0, None, None
        return 1, y, y_hat

if __name__ == "__main__":

    x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    mlogr = MyLogisticRegression(theta)
    print(mlogr.gradient_(x, y, lambda_=0.0))
