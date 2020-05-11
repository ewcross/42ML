# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    l2_reg.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/11 09:51:11 by ecross            #+#    #+#              #
#    Updated: 2020/05/11 10:11:55 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def l2(theta):
    if type(theta) != np.ndarray or theta.ndim > 2:
        print('theta should be a 1d ndarray')
        return None
    theta = theta.reshape(-1,)
    theta[0] = 0
    return np.dot(theta, theta)

if __name__ == "__main__":

    th = np.array([2, 14, -13, 5, 12, 4, -19])
    print(l2(th))
    th = np.array([3,0.5,-6])
    print(l2(th))
