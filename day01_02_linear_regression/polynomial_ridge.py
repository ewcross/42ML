# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_ridge.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/12 08:18:58 by ecross            #+#    #+#              #
#    Updated: 2020/05/12 11:50:16 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from my_linear_regression import MyLinearRegression as MLR
from ridge import MyRidge as MRG
sys.path.insert(1, '/Users/elliotcross/Documents/42/python/bootcamp_ml/tools')
from add_polynomial_features import add_polynomial_features
from normalisation import minmax
from data_splitter import data_splitter

def plot(x, y, y_hat, original_features):
    for i in range(original_features.shape[1]):
        plt.plot(x_test[:, i:i + 1], y_test, 'o')
        plt.title(data.columns[i])
        plt.plot(x_test[:, i:i + 1], y_hat, 'ro', markersize=3)
        plt.show()

data = pd.read_csv("../subjects/day02/resources/spacecraft_data.csv")
features = np.array(data[["Age", "Thrust_power", "Terameters"]])
y = np.array(data["Sell_price"]).reshape(-1,1)

#first create polynomial feature array, of degree 3

new_features = add_polynomial_features(features, 3)
    
#then normalise it

for i in range(new_features.shape[1]):
    new_features[:, i:i + 1] = minmax(new_features[:, i:i + 1])

#then shuffle into training and testing sets

x_train, x_test, y_train, y_test = data_splitter(new_features, y, 0.5)

for i in range(3):
    plt.plot(x_train[:, i:i + 1], y_train, 'go')
    plt.title(data.columns[i])
    plt.plot(x_test[:, i:i + 1], y_test, 'ro', markersize=3)
    plt.show()

#initialise thetas as array with feature number + 1 zeros
thetas = np.zeros(new_features.shape[1] + 1)

#should be able to use same alpha and cycle number for all, as same data

#carry out linear regression on training data

cost_list = []

mlr = MLR(thetas, alpha=0.1, n_cycle=400)
mlr.fit_(x_train, y_train)
y_hat = mlr.predict_(x_test)[1]
cost_list.append(mlr.mse_(y_test, y_hat))
plot(x_test, y_test, y_hat, features)

#carry out 9 ridge regressions on training data, with lambda from 0.1 to 0.9

mrg = MRG(thetas, alpha=0.1, n_cycle=400)
for i in range(1, 10):
    mrg.lambda_ = i / 10
    mrg.thetas = thetas
    plt.title('lambda = ' + str(i / 10))
    mrg.fit_(x_train, y_train)
    y_hat = mrg.predict_(x_test)[1]
    cost_list.append(mlr.mse_(y_test, y_hat))
    #plot(x_test, y_test, y_hat, features)

plt.bar(range(0, 10), cost_list)
plt.show()
