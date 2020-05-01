# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/01 20:22:40 by ecross            #+#    #+#              #
#    Updated: 2020/05/01 21:42:29 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

from vec_gradient import gradient
from plot import plot

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

    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1])
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter = 150000)
    print(theta1)
    plot(x, y, theta1)
