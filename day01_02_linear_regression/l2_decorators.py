# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    l2_decorators.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/11 12:18:40 by ecross            #+#    #+#              #
#    Updated: 2020/05/11 16:15:17 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

from my_linear_regression import MyLinearRegression as MLR

def add_l2_to_mse(cost_function):
    """
    decorator function which returns a decorated version of a 
    cost function, which adds an L-squared term to the cost value
    returned by the original cost function
    """
    def get_regularised_cost(*args):
        if len(args) != 3:
            print('please just supply y and y_hat')
            return None
        cost = cost_function(args[0], args[1], args[2])
        if cost is None:
            print('error calculating cost before regularisation')
            return None
        if args[0].thetas.ndim > 2:
            print('theta must be a 1d ndarray')
            return None
        theta = np.array(args[0].thetas)
        theta = theta.reshape(-1,)
        theta[0] = 0
        return (cost + ((args[0].lambda_ / args[1].shape[0]) 
                * np.dot(theta, theta)))
    return get_regularised_cost

def add_l2_to_gradient(gradient_function):
    """
    decorator function which returns a decorated version of a
    gradient function, which adds an L-squared term to the gradient
    vector (each element) returned by the original function
    """
    def get_regularised_gradient(*args):
        if len(args) != 3:
            print('please just supply x and y vectors')
            return None
        gradient = gradient_function(args[0], args[1], args[2])
        if gradient is None:
            print('error calculating gradient before regularisation')
            return None
        if args[0].thetas.ndim > 2:
            print('theta must be a 1d ndarray')
            return None
        theta = np.array(args[0].thetas)
        theta = theta.reshape(-1,)
        theta[0] = 0
        lambda_theta = theta * args[0].lambda_
        return gradient + (lambda_theta / args[1].shape[0])
    return get_regularised_gradient
