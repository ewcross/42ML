# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    normalisation.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/04 22:05:00 by ecross            #+#    #+#              #
#    Updated: 2020/05/04 22:19:25 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def zscore(x):

    """Computes the normalized version of a non-empty
    numpy.ndarray using the z-score standardisation - reshapes
    the array to 1d if not already"""

    if x.size == 0:
        return None
    if x.ndim > 1:
        x.reshape(1, )
    mean = np.mean(x)
    std = np.std(x)
    x = x - mean
    return x / std

def minmax(x):
    """Computes the normalized version of a non-empty
    numpy.ndarray using the min-max standardisation - reshapes
    the array to 1d if not already"""

    if x.size == 0:
        return None
    if x.ndim > 1:
        x.reshape(1, )
    r = np.amax(x) - np.amin(x)
    x = x - np.amin(x)
    return x / r

if __name__ == "__main__":
    
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    print()
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(zscore(Y))
    print()
    print(minmax(X))
    print()
    print(minmax(Y))
