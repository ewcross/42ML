# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/01 19:53:44 by ecross            #+#    #+#              #
#    Updated: 2020/05/01 21:32:35 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept(x):
    """takes a non-empty numpy.ndarray with 1 dimension (vector)
    and returns a 2D matrix with a column of 1's and a column
    containing the original vector"""

    #return np.array([np.ones(x.size), x]).transpose()
    if x.ndim == 1:
        x = x[:, None]
    return np.insert(x, 0, 1, axis=1)

def check_size_and_shape(y, y_hat):
    if y.size == 0 or y_hat.size == 0:
        return 0, None, None
    if y.ndim > 1:
        y = y.reshape((y.size,))
    if y_hat.ndim > 1:
        y_hat = y_hat.reshape((y_hat.size,))
    if y.shape != y_hat.shape:
        return 0, None, None
    return 1, y, y_hat

def gradient(x, y, theta):
    """computes a 1d vector of the gradients of the cost function - the
    partial derivitive with respect to each theta"""

    #add column of 1s to right side of x value vector
    #number of theta values must match number of features (n in xn)
    #get 1d vector of y_hat (predicted) values
    #check that y_hat (calculated from x) and y are compatible
    #dot product of transpose of x with y_hat - y vector gives vector of gradients

    x = add_intercept(x)
    if theta.shape != (x.shape[1],):
        return None
    y_hat = np.dot(x, theta)
    check, y, h = check_size_and_shape(y, y_hat)
    if not check:
        return None
    y_hat = y_hat - y
    return np.dot(x.transpose(), y_hat) * (1 / y.size)

if __name__ == "__main__":

    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    theta1 = np.array([1, -0.4])
    print("theta:")
    print(gradient(x, y, theta1))
