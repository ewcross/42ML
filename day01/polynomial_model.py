# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_model.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/05 18:10:15 by ecross            #+#    #+#              #
#    Updated: 2020/05/05 18:37:09 by ecross           ###   ########.fr        #
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
        last_col = new[:, -1][:, None]
        new = np.concatenate((new, last_col * x), axis=1)
    return new

if __name__ == "__main__":

    x = np.arange(10)
    print(add_polynomial_features(x, 2))
