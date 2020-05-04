# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/01 20:22:40 by ecross            #+#    #+#              #
#    Updated: 2020/05/04 20:54:53 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from vec_gradient import gradient
from plot import plot
from cost import half_mse_
from prediction import predict_

def check(x, y, theta, alpha, max_iter):
    if (type(x) != np.ndarray or type(y) != np.ndarray or 
            type(theta) != np.ndarray):
        print("x, y and theta must be numpy ndarrays")
        return 0
    if not isinstance(alpha, float):
        print("alpha must be float")
        return 0
    if not isinstance(max_iter, int):
        print("max_iter must be int")
        return 0
    return 1

def fit_(x, y, theta, alpha, max_iter):
    if not check(x, y, theta, alpha, max_iter):
        return None
    for i in range(max_iter):
        nabla = gradient(x, y, theta)
        if nabla is None:
            return None
        theta = theta - (nabla * alpha)
    return theta

if __name__ == "__main__":

    x, y = make_regression(n_samples=10, n_features=1, noise=10)
    theta = np.array([1, 1])
    plt.plot(x, y, 'o')
   
    theta1 = fit_(x, y, theta, 0.1, 100)
    print(theta1)
    plt.plot(x, (theta1[1] * x + theta1[0]), '-r')
    plt.show()
