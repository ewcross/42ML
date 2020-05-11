# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ridge.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/11 12:53:07 by ecross            #+#    #+#              #
#    Updated: 2020/05/11 16:10:00 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

from my_linear_regression import MyLinearRegression as MLR
from l2_decorators import add_l2_to_mse, add_l2_to_gradient 

class MyRidge(MLR):
    """
    ridge regression class to perform linear regression with
    regularisation using l2 decorators
    """
    def __init__(self, thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
        if not self.check_init(thetas, alpha, n_cycle, lambda_):
            raise ValueError
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = thetas
        self.lambda_ = lambda_
    
    def check_init(self, theta, alpha, n_cycle, lambda_):
        if type(theta) != np.ndarray:
            print("theta must be 1D numpy array")
            return 0
        if type(alpha) != float and type(alpha) != int:
            print("alpha must be float or int")
            return 0
        if type(n_cycle) != int:
            print("n_cycle must be int")
            return 0
        if type(lambda_) != float:
            print("lamnda must be float")
            return 0
        return 1

    @add_l2_to_gradient
    def gradient(self, x, y):
        return MLR.gradient(self, x, y)

    @add_l2_to_mse
    def mse_(self, y, y_hat):
        return MLR.mse_(self, y, y_hat)


if __name__ == "__main__":

    x = np.array([[ -6, -7, -9],[ 13, -2, 14],[ -7, 14, -1],[-8,-4, 6],[-5,-9, 6],[ 1, -5, 11],[9,-11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    myr = MyRidge(theta, lambda_=1.)
    print(myr.gradient(x, y))
