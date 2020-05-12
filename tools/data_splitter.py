# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_splitter.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/06 10:34:27 by ecross            #+#    #+#              #
#    Updated: 2020/05/06 11:19:46 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def data_splitter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a 
    training and a test set, while respecting the given proportion 
    of examples to be kept in the traning set
    returns: (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarrays"""

    if proportion <= 0 or proportion >= 1:
        print('please give a valid proportion between 0 and 1')
        return (None,) * 4
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    new = np.concatenate((y, x), axis=1)
    np.random.shuffle(new)
    line = math.ceil(new.shape[0] * proportion)
    if line == new.shape[0]:
        line -= 1
    return (new[:line, 1:], new[line:, 1:], new[:line, :1], new[line:, :1])

if __name__ == "__main__":

    x = np.array([1, 42, 300, 10, 59])
    x1 = np.array([[1, 1], [42, 42], [300, 300], [10, 10], [59, 59]])
    y = np.array([11.11, 42.42, 33.33, 10.10, 59.59])
    print(x1)
    print(y)
    print("------------------------------")
    x_train, x_test, y_train, y_test = data_splitter(x1, y, 0.5)
    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)
