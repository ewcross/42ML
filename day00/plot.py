# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 18:47:32 by ecross            #+#    #+#              #
#    Updated: 2020/04/30 18:55:15 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
    plt.plot(x, y, 'o')
    plt.plot(x, (theta[1] * x + theta[0]), '-')
    plt.show()

if __name__ == "__main__":

    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    theta2 = np.array([3, 0.3])
    plot(x, y, theta2)
