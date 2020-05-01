# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cost.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 20:04:08 by ecross            #+#    #+#              #
#    Updated: 2020/05/01 11:21:02 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

from prediction import predict_

def cost_elem_(y, y_hat):
    if y.ndim == 1:
        y = y[:, None]
    if y_hat.ndim == 1:
        y_hat = y_hat[:, None]
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    x = np.array([])
    const = 1 / (2 * y.shape[0])
    for orig, hat in zip(y, y_hat):
        x = np.append(x, const * (hat - orig)**2)
    return(x)

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

def half_mse_(y, y_hat):
    check, y, y_hat = check_size_and_shape(y, y_hat)
    if not check:
        return None
    y = y_hat - y
    return np.dot(y, y) / (2 * y.size)

def mse_(y, y_hat):
    check, y, y_hat = check_size_and_shape(y, y_hat)
    if check == None:
        return None
    y = y_hat - y
    return np.dot(y, y) / y.size

def rmse_(y, y_hat):
    check, y, y_hat = check_size_and_shape(y, y_hat)
    if check == None:
        return None
    y = y_hat - y
    return math.sqrt(np.dot(y, y) / y.size)

def mae_(y, y_hat):
    check, y, y_hat = check_size_and_shape(y, y_hat)
    if check == None:
        return None
    y = y_hat - y
    y = np.absolute(y)
    ones = np.ones(y.size)
    return np.dot(y, ones) / y.size

def r2score_(y, y_hat):
    check, y, y_hat = check_size_and_shape(y, y_hat)
    if check == None:
        return None
    mean_y = np.mean(y)
    bottom = y_hat - mean_y
    top = y_hat - y
    return 1 - (np.dot(top, top) / np.dot(bottom, bottom))

if __name__ == "__main__":

    y = np.array([0, 15, -9, 7])
    y_hat = np.array([2, 14, -13, 5])
    print(r2score_(y, y_hat))
