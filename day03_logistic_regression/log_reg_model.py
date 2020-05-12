# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_reg_model.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/09 15:24:46 by ecross            #+#    #+#              #
#    Updated: 2020/05/12 11:48:21 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

sys.path.insert(1, '/Users/elliotcross/Documents/42/python/bootcamp_ml/tools')
from my_logistic_regression import MyLogisticRegression as MLogR
from confusion_matrix import confusion_matrix_
from data_splitter import data_splitter
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_
from add_polynomial_features import add_polynomial_features
from normalisation import minmax

def select_label(y, label):
    new = [0 if i[0] != label else 1 for i in y]
    return np.array(new).reshape(-1, 1)

planets = pd.read_csv("../subjects/day03/resources/solar_system_census_planets.csv")
people = pd.read_csv("../subjects/day03/resources/solar_system_census.csv")
origins = np.array(planets[["Origin"]]).reshape(-1,1)
X = np.array(people[["height", "weight", "bone_density"]]).reshape(-1, 3)

#normalise the data
X = np.concatenate((minmax(X[:, :1]), minmax(X[:, 1:2]), minmax(X[:, 2:3])), axis=1)

#split data into training and testing sets
X_train, X_test, origins_train, origins_test = data_splitter(X, origins, 0.5)

thetas = np.zeros(4)

mlogr = MLogR(thetas, alpha=0.9, n_cycle=2000)
for i in range(4):
    mlogr.thetas = thetas
    origins_train_label = select_label(origins_train, i)
    origins_test_label = select_label(origins_test, i)
    #mlogr.fit_(X_train, origins_train_label)
    plt.title(('label: ' + str(i)))
    mlogr.plot_convergence(X_train, origins_train_label)
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

labels = np.array([get_biggest_index(i) for i in results]).reshape(-1, 1)

for i in range(3):
    plt.plot(X_test[:, i:i + 1], origins_test, 'o')
    plt.plot(X_test[:, i:i + 1], labels, 'ro', markersize=3)
    plt.title(people.columns[i + 1])
    plt.show()

print(confusion_matrix_(origins_test, labels, df_option=True))

for i in range(4):
    y_labelled = select_label(origins_test, i)
    y_hat_labelled = select_label(labels, i)
    print(f'planet {i} accuracy        :', accuracy_score_(y_labelled, y_hat_labelled))
    print(f'planet {i} precision (fps)  :', precision_score_(y_labelled, y_hat_labelled))
    print(f'planet {i} recall (fns)  :', recall_score_(y_labelled, y_hat_labelled))
    print(f'planet {i} f1        :', f1_score_(y_labelled, y_hat_labelled))
    print('-----------------------------')
