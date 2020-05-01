# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    gradient.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/01 15:35:49 by ecross            #+#    #+#              #
#    Updated: 2020/05/01 18:44:53 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

from prediction import predict_

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

def simple_gradient(x, y, theta):
    h = predict_(x, theta)
    if h is None:
        return None
    check, y, h = check_size_and_shape(y, h)
    if not check:
        return None
    #x[:, :1] is the first column of x values (as x here does not have column of 1s)
    return np.array([np.sum(h - y) / y.size, np.sum((h - y) * x[:, :1]) / y.size])

if __name__ == "__main__":

    x = np.array([[12.4956442, 21.5007972, 31.5527382], [48.9145838, 57.5088733, 59.988272]])
    y = np.array([10.4956442, 19.500744])
    theta1 = np.array([1, -0.4, -2, 1.2])
    print("theta: ", simple_gradient(x, y, theta1))
