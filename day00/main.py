# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/04 12:33:11 by ecross            #+#    #+#              #
#    Updated: 2020/05/04 12:46:15 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from my_linear_regression import MyLinearRegression as MLR

if __name__ == "__main__":

    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    theta = np.array([1, 1])
    
    #mlr = MLR(theta, 0.001, 500)
    #theta1 = mlr.fit_(x, y)
    #plt.plot(x, y, 'o')
    #plt.plot(x, (theta1[1] * x + theta1[0]), '-r')
    
    mlr = MLR(theta, 0.5, 1100)
    theta1 = mlr.fit_(x, y)
    plt.plot(x, y, 'o')
    plt.plot(x, (theta1[1] * x + theta1[0]), '-g')
    
    plt.show()
