# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multivariate_linear_model.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/05 12:23:37 by ecross            #+#    #+#              #
#    Updated: 2020/05/05 17:43:39 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_linear_regression import MyLinearRegression as MLR

def plot(mlr, x, y, xl):
    th = mlr.fit_(x, y)
    print(th)
    plt.plot(x, y, "ob")
    plt.xlabel(xl)
    plt.plot(x, (th[0] + (th[1] * x)), 'og')
    plt.show()

def plot_th(x, y, th, xl):
    plt.plot(x, y, "ob")
    plt.xlabel(xl)
    y = th[0] + (th[1] * x) + th[2] + th[3]
    plt.plot(x, y, "o", markersize=2)
    plt.show()

data = pd.read_csv("../subjects/day02/resources/spacecraft_data.csv")
age = np.array(data["Age"]).reshape(-1,1)
tp = np.array(data["Thrust_power"]).reshape(-1,1)
tm = np.array(data["Terameters"]).reshape(-1,1)
features = np.array(data[["Age", "Thrust_power", "Terameters"]])
y = np.array(data["Sell_price"]).reshape(-1,1)

thetas = np.array([1, 1])
plt.ylabel("price")

#mlr_age = MLR(thetas, alpha=0.01, n_cycle=4000)
#plot(mlr_age, age, y, "age")

#mlr_thrust = MLR(thetas, alpha=0.00001, n_cycle=30)
#plot(mlr_thrust, tp, y, "thrust power")

#mlr_tm = MLR(thetas, alpha=0.00022, n_cycle=76000)
#plot(mlr_tm, tm, y, "terameters")

thetas = np.array([1, 1, 1, 1])

mlr_multi = MLR(thetas, alpha=0.00009, n_cycle=100)
#mlr_multi.plot_cost_change(features, y)
th = mlr_multi.fit_(features, y)
print(th)
y_hat = th[0]
i = 1
while i < 4:
    y_hat += th[i] * features[:, i - 1:i]
    i += 1

plt.plot(age, y, "ob")
plt.xlabel("age")
plt.plot(age, y_hat, "o", markersize=2)
plt.show()
plt.plot(tp, y, "ob")
plt.xlabel("thrust power")
plt.plot(tp, y_hat, "o", markersize=2)
plt.show()
plt.plot(tm, y, "ob")
plt.xlabel("terameters")
plt.plot(tm, y_hat, "o", markersize=2)
plt.show()
