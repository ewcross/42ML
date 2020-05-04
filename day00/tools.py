# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tools.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 18:08:07 by ecross            #+#    #+#              #
#    Updated: 2020/05/04 20:40:42 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept(x):
    """takes a non-empty numpy.ndarray with 1 dimension (vector)
    and returns a 2D matrix with a column of 1's and a column
    containing the original vector"""

    #return np.array([np.ones(x.size), x]).transpose()
    if x.ndim == 1:
        x = x[:, None]
    return np.insert(x, 0, 1, axis=1)

if __name__ == "__main__":
    
    x = np.arange(1,10).reshape((3, 3))
    print(add_intercept(x))
