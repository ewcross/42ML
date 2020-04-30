# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 17:29:40 by ecross            #+#    #+#              #
#    Updated: 2020/04/30 18:23:52 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

from tools import add_intercept

def predict_(x, theta):
    """computes the vector of prediction y_hat
    from two non-empty numpy.ndarray using h(x) = th0 + th1(x)."""
    
    x = add_intercept(x)
    return np.dot(x, theta)

if __name__ == "__main__":
    
    x = np.arange(1,6)
    theta1 = np.array([5, 0])
    print(predict_(x, theta1))

    theta2 = np.array([0, 1])
    print(predict_(x, theta2))

    theta3 = np.array([5, 3])
    print(predict_(x, theta3))

    theta4 = np.array([-3, 1])
    print(predict_(x, theta4))
