# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_cost_reg.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/11 10:05:25 by ecross            #+#    #+#              #
#    Updated: 2020/05/11 10:33:14 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def check_size_and_shape(y, y_hat):
    if type(y) != np.ndarray or type(y_hat) != np.ndarray:
        return 0, None, None
    if y.size == 0 or y_hat.size == 0:
        return 0, None, None
    if y.ndim > 1:
        y = y.reshape((y.size,))
    if y_hat.ndim > 1:
        y_hat = y_hat.reshape((y_hat.size,))
    if y.shape != y_hat.shape:
        return 0, None, None
    return 1, y, y_hat

def l2(theta):
    if type(theta) != np.ndarray or theta.ndim > 2:
        print('theta should be a 1d ndarray')
        return None
    theta = theta.reshape(-1,)
    theta[0] = 0
    return np.dot(theta, theta)

def reg_cost_(y, y_hat, theta, lambda_):
    if type(lambda_) != float and type(lambda_) != int:
        print('lambda value must be a float or int')
        return None
    if type(theta) != np.ndarray or theta.ndim > 2:
        print('wrong format for theta')
        return None
    theta = theta.reshape(-1, 1)
    check, y, y_hat = check_size_and_shape(y, y_hat)
    if check == 0:
        print('input vectors do not match')
        return None
    y = y_hat - y
    return (np.dot(y, y) / (2 * y.size)) + ((lambda_ * l2(theta)) / (2 * y.size))

if __name__ == "__main__":

    y = np.array([2, 14, -13, 5, 12, 4, -19])
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20])
    theta = np.array([1, 2.5, 1.5, -0.9])
    print(reg_cost_(y, y_hat, theta, .5))
    print(reg_cost_(y, y_hat, theta, .05))
    print(reg_cost_(y, y_hat, theta, .9))
