# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_train.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/05 18:33:34 by ecross            #+#    #+#              #
#    Updated: 2020/05/05 21:31:20 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd

from my_linear_regression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features

data = pd.read_csv("../subjects/day01/resources/are_blue_pills_magics.csv")
x = np.array(data["Micrograms"]).reshape(-1,1)
y = np.array(data["Score"]).reshape(-1,1)

new = add_polynomial_features(x, 10)

thetas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cost_values = []

degree = 2
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.002, n_cycle=1500)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 3
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.00005, n_cycle=4000)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 4
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.0000001, n_cycle=50)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 5
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.00000005, n_cycle=15)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 6
mlr = MLR(np.array(thetas[:degree + 1]), alpha=0.0000000001, n_cycle=30)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 7
mlr = MLR(np.array(thetas[:degree + 1]), alpha=5e-12, n_cycle=15)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 8
mlr = MLR(np.array(thetas[:degree + 1]), alpha=5e-14, n_cycle=40)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 9
mlr = MLR(np.array(thetas[:degree + 1]), alpha=1e-15, n_cycle=55)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

degree = 10
mlr = MLR(np.array(thetas[:degree + 1]), alpha=8e-17, n_cycle=20)
mlr.fit_(new[:, :degree], y)
cost_values.append(mlr.mse_(y, mlr.predict_(new[:, :degree])[1]))

print(cost_values)
