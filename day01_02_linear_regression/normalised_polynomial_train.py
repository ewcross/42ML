# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    normalised_polynomial_train.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/06 11:41:29 by ecross            #+#    #+#              #
#    Updated: 2020/05/12 11:49:56 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from my_linear_regression import MyLinearRegression as MLR
sys.path.insert(1, '/Users/elliotcross/Documents/42/python/bootcamp_ml/tools')
from polynomial_model import add_polynomial_features
from normalisation import minmax

data = pd.read_csv("../subjects/day01/resources/are_blue_pills_magics.csv")
x_train = np.array(data["Micrograms"]).reshape(-1,1)
y_train = np.array(data["Score"]).reshape(-1,1)

x_test = np.array([5, 4.3, 2, 2, 5, 6, 3.5]).reshape(-1, 1)
y_test = np.array([39, 52, 70, 58, 50, 32, 62]).reshape(-1, 1)


new_train = add_polynomial_features(x_train, 10)
#normalise new_train
for i in range(10):
    new_train[:, i] = minmax(new_train[:, i])

#for plotting of polynomial curves - cotinuous data set over range of original data
#then add polynomial features and normalise
continuous_x = np.arange(1, 7.01, 0.01).reshape(-1, 1)
x_ = add_polynomial_features(continuous_x, 10)
for i in range(10):
    x_[:, i] = minmax(x_[:, i])

thetas = np.ones(11).reshape(-1, 1)

cost_values = []
thetas_list = []
mlr = MLR(thetas, alpha=0.009, n_cycle=5000)
for degree in range(2, 11):
    mlr.thetas = thetas[:degree + 1]
    thetas_list.append(mlr.fit_(new_train[:, :degree], y_train))
    cost_values.append(mlr.mse_(y_train, mlr.predict_(new_train[:, :degree])[1]))

i = 2
for elem in thetas_list:
    mlr.thetas = elem
    y_hat = mlr.predict_(x_[:, :i])[1]
    plt.plot(continuous_x, y_hat, '--')
    plt.title(str(degree))
    plt.title(('degree = ' + str(i) + ' cost: ' + str(cost_values[i - 2])))
    plt.plot(x_train, y_train, 'go')
    plt.plot(x_test, y_test, 'ro')
    plt.show()
    i += 1
