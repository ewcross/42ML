# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/04 11:38:14 by ecross            #+#    #+#              #
#    Updated: 2020/05/05 19:00:17 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression:

    def __init__(self, thetas, alpha=0.001, n_cycle=1000):
        if not self.check_init(thetas, alpha, n_cycle):
            del self
            return
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = thetas

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

        """carries out linear regression given a set of data features, xn,
        training output values, y, and initial theta values (self.thetas),
        returning a vector containing new theta values"""

        if type(x) != np.ndarray or type(y) != np.ndarray:
            print("x, y and theta must be numpy ndarrays")
            return None
        for i in range(self.n_cycle):
            nabla = self.gradient(x, y)
            if nabla is None:
                return None
            self.thetas = self.thetas - (nabla * self.alpha)
        return self.thetas

    def plot_convergence(self, x, y):

        """carries out linear regression given a set of data features, xn,
        training output values, y, and initial theta values (self.thetas),
        returning a vector containing new theta values - this version plots
        the change in cost function with convergence cycles, and displays this"""

        if type(x) != np.ndarray or type(y) != np.ndarray:
            print("x, y and theta must be numpy ndarrays")
            return None
        j_values = []
        for i in range(self.n_cycle):
            nabla = self.gradient(x, y)
            if nabla is None:
                return None
            self.thetas = self.thetas - (nabla * self.alpha)
            j_values.append(self.mse_(y, self.predict_(x)[1]))
        j = np.array(j_values)
        epoc = np.array(np.arange(self.n_cycle))
        plt.plot(epoc, j, 'o')
        plt.show()
        return self.thetas

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

    def gradient(self, x, y):

        """computes a 1d vector of the gradients of the cost function, the
        partial derivitive with respect to each theta, for given theta values"""
    
        #add column of 1s to right side of x value vector
        #number of theta values must match number of features (n in xn)
        #get 1d vector of y_hat (predicted) values
        #check that y_hat (calculated from x) and y are compatible
        #dot product of transpose of x with (y_hat - y) vector gives vector of gradients
    
        x, y_hat = self.predict_(x)
        if x is None or y_hat is None:
            return None
        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            return None
        y_hat = y_hat - y
        return np.dot(x.transpose(), y_hat) * (1 / y.size)

    def predict_(self, x):

        """generates a vector of predicted y values given input data and thetas,
        also returning the x vector with 1s column added"""

        x = self.add_intercept(x)
        if self.thetas.ndim > 1:
            self.thetas = self.thetas.reshape((self.thetas.size,))
        if self.thetas.shape != (x.shape[1],):
            print("mismatch between theta vector size, and input features")
            return None, None
        return x, np.dot(x, self.thetas)

    def add_intercept(self, x):

        """takes a non-empty numpy.ndarray and returns a 2D matrix
        with a column of 1's concatenated to the right hand side
        of the  original vector"""

        #return np.array([np.ones(x.size), x]).transpose()
        if x.ndim == 1:
            x = x[:, None]
        return np.insert(x, 0, 1, axis=1)

    def mse_elem_(y, y_hat):

        """generates the value of the cost function for for each element of
        a set of predicted and actual values, y_hat and y,
        using mean squared error"""

        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            return None
        return (y_hat - y) / y.size

    def mse_(self, y, y_hat):

        """generates the value of the cost function for a set of predicted
        and actual values, y_hat and y, using mean squared error"""

        check, y, y_hat = self.check_size_and_shape(y, y_hat)
        if not check:
            return None
        y = y_hat - y
        return np.dot(y, y) / y.size
