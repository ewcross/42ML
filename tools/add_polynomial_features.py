# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    add_polynomial_features.py                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/11 09:04:59 by ecross            #+#    #+#              #
#    Updated: 2020/05/11 09:21:43 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values
    up to the power given as argument 'power'"""

    if power < 1:
        print("please give a power larger than 0")
        return None
    if x.ndim == 1:
        x = x[:, None]
    new = x
    for i in range(1, power):
        last_cols = new[:, -x.shape[1]:]
        if last_cols.ndim == 1:
            last_cols = last_cols[:, None]
        new = np.concatenate((new, last_cols * x), axis=1)
    return new

if __name__ == "__main__":

    x = np.arange(1,11).reshape(5, 2)
    print(add_polynomial_features(x, 1))
    print(add_polynomial_features(x, 3))
    print(add_polynomial_features(x, 4))
    print(add_polynomial_features(x, 5))
