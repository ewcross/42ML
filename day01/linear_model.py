# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_model.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/04 21:03:21 by ecross            #+#    #+#              #
#    Updated: 2020/05/04 21:55:10 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from my_linear_regression import MyLinearRegression as MyLR

data = pd.read_csv("../subjects/day01/resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)
thetas = np.array([1, 1])
#plt.plot(Xpill, Yscore, 'o')

mlr = MyLR(thetas, alpha=0.05, n_cycle=5000)
#th = mlr.fit_(Xpill, Yscore)
#print(th)
#plt.plot(Xpill, (th[1] * Xpill + th[0]), '-r')
#plt.show()

for j in range(80, 100, 5):
    res = []
    for i in range(-11, -7, 1):
        mlr.thetas = np.array([j, i])
        dummy, y_hat = mlr.predict_(Xpill)
        res.append(mlr.mse_(Yscore, y_hat))
    np.array(res)
    plt.plot(np.arange(-11, -7), res)
plt.show()

for j in range(80, 100, 5):
    res = []
    for i in range(-11, -7, 1):
        mlr.thetas = np.array([j, i])
        dummy, y_hat = mlr.predict_(Xpill)
        res.append(mean_squared_error(Yscore, y_hat))
    np.array(res)
    plt.plot(np.arange(-11, -7), res, '-r')
plt.show()
