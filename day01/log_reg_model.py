# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_reg_model.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/09 15:24:46 by ecross            #+#    #+#              #
#    Updated: 2020/05/09 21:26:00 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MLogR
from data_splitter import data_splitter

def select_label(y, label):
    new = [0 if i[0] != label else 1 for i in y]
    return np.array(new).reshape(-1, 1)

planets = pd.read_csv("../subjects/day03/resources/solar_system_census_planets.csv")
people = pd.read_csv("../subjects/day03/resources/solar_system_census.csv")
origins = np.array(planets[["Origin"]]).reshape(-1,1)
X = np.array(people[["height", "weight", "bone_density"]]).reshape(-1, 3)
X_train, X_test, origins_train, origins_test = data_splitter(X, origins, 0.5)
thetas = np.zeros(4)

mlogr = MLogR(thetas, alpha=0.0001, n_cycle=1000)
for i in range(4):
    mlogr.thetas = thetas
    origins_train_label = select_label(origins_train, i)
    origins_test_label = select_label(origins_test, i)
    mlogr.fit_(X_train, origins_train_label)
    if i == 0:
        results = mlogr.predict_(X_test).reshape(-1, 1)
    if i > 0:
        results = np.concatenate((results, mlogr.predict_(X_test).reshape(-1, 1)), axis=1)

def get_biggest_index(row):
    biggest = -1
    index = -1
    i = 0
    for elem in row:
        if elem > biggest:
            biggest = elem
            index = i
        i += 1
    return index

labels = [get_biggest_index(i) for i in results]
for planet, label in zip(origins_test, labels):
    print(planet, ' --> ', label)
