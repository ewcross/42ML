# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_train.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/05 18:33:34 by ecross            #+#    #+#              #
#    Updated: 2020/05/12 11:50:40 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from my_linear_regression import MyLinearRegression as MLR
sys.path.insert(1, '/Users/elliotcross/Documents/42/python/bootcamp_ml/tools')
from polynomial_model import add_polynomial_features

data = pd.read_csv("../subjects/day01/resources/are_blue_pills_magics.csv")
x = np.array(data["Micrograms"]).reshape(-1,1)
y = np.array(data["Score"]).reshape(-1,1)
#plt.plot(x, y, 'o')
#plt.show()

x_test = np.array([5, 4.3, 2, 2, 5, 6, 3.5]).reshape(-1, 1)
y_test = np.array([39, 52, 70, 58, 50, 32, 62]).reshape(-1, 1)
plt.plot(x_test, y_test, 'o')

new = add_polynomial_features(x, 10)

thetas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cost_values = []

#for plotting of polynomial curves - cotinuous data set over range of original data
continuous_x = np.arange(1, 7.01, 0.01).reshape(-1, 1)
x_ = add_polynomial_features(continuous_x, 10)

degree = 2
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.002, n_cycle=1500)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 3
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.00005, n_cycle=4000)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 4
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.0000001, n_cycle=50)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 5
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.00000005, n_cycle=15)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 6
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.0000000001, n_cycle=30)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 7
mlr = MLR(np.array(thetas[:degree + 1]), alpha=5e-12, n_cycle=15)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 8
mlr = MLR(np.array(thetas[:degree + 1]), alpha=5e-14, n_cycle=40)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 9
mlr = MLR(np.array(thetas[:degree + 1]), alpha=1e-15, n_cycle=55)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

degree = 10
mlr = MLR(np.array(thetas[:degree + 1]), alpha=8e-17, n_cycle=20)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))
y_hat = mlr.predict_(x_[:, :degree])[1]
plt.plot(x, y, 'o')
plt.plot(continuous_x, y_hat, '--')
plt.title(str(degree))
plt.show()

#print(cost_values)
#cost_x = np.arange(2, 11)
#cost_y = np.array(cost_values)
#plt.plot(cost_x, cost_y, 'o')
#plt.show()
