# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 18:47:32 by ecross            #+#    #+#              #
#    Updated: 2020/05/01 21:03:15 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

from prediction import predict_
from cost import half_mse_

def plot(x, y, theta):
    plt.plot(x, y, 'o')
    plt.plot(x, (theta[1] * x + theta[0]), '-')
    plt.show()

def plot_with_cost(x, y, theta):
    plt.plot(x, y, 'o')
    plt.plot(x, (theta[1] * x + theta[0]), '-')
    y_hat = predict_(x, theta)
    cost = half_mse_(y, y_hat)
    plt.plot((x, x), (y, y_hat), '--r')
    plt.title(("Cost: " + str(round(cost, 6))))
    plt.show()

if __name__ == "__main__":
    
    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    theta= np.array([12, 0.8])
    plot_with_cost(x, y, theta)
