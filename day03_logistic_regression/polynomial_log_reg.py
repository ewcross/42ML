# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_log_reg.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/09 15:24:46 by ecross            #+#    #+#              #
#    Updated: 2020/05/12 11:48:28 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import sys

sys.path.insert(1, '/Users/elliotcross/Documents/42/python/bootcamp_ml/tools')
from my_logistic_regression import MyLogisticRegression as MLogR
from confusion_matrix import confusion_matrix_
from data_splitter import data_splitter
from other_metrics import f1_score_
from add_polynomial_features import add_polynomial_features
from normalisation import minmax

def select_label(y, label):
    new = [0 if i[0] != label else 1 for i in y]
    return np.array(new).reshape(-1, 1)

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

def get_biggest_value(row):
    biggest = -1
    index = -1
    i = 0
    for elem in row:
        if elem > biggest:
            biggest = elem
            index = i
        i += 1
    return biggest

planets = pd.read_csv("../subjects/day03/resources/solar_system_census_planets.csv")
people = pd.read_csv("../subjects/day03/resources/solar_system_census.csv")
Y = np.array(planets[["Origin"]]).reshape(-1,1)
X = np.array(people[["height", "weight", "bone_density"]]).reshape(-1, 3)

#add polynomial features to a degree of 3
X = add_polynomial_features(X, 3)

#normalise the data
for i in range(X.shape[1]):
    X[:, i:i + 1] = minmax(X[:, i:i + 1])

#split data into training and testing sets
X_train, X_test, Y_train, Y_test = data_splitter(X, Y, 0.5)

thetas = np.zeros(X.shape[1] + 1)

mlogr = MLogR(thetas, alpha=0.02, n_cycle=2000, penalty='l2')
#for each category (0, 1, 2, 3) carry out logistic regression
#and concatenate the 4 predicted probability vectors, giving a matrix
#with the probablitiy that each element belongs to each category
f1_scores = []
for l in range(0, 100, 10):
    for i in range(4):
        mlogr.thetas = thetas
        mlogr.lambda_ = float(l / 10)
        Y_train_label = select_label(Y_train, i)
        mlogr.fit_(X_train, Y_train_label)
        if i == 0:
            results = mlogr.predict_(X_test).reshape(-1, 1)
        if i > 0:
            results = np.concatenate((results, mlogr.predict_(X_test).reshape(-1, 1)), axis=1)

    #choose the most probable category for each element and condense into a vector
    final_labels = np.array([get_biggest_index(i) for i in results]).reshape(-1, 1)
    
    #plot the test results and predicted results together, one graph for each feature
    #for i in range(4):
    #    plt.plot(X_test[:, i:i + 1], Y_test, 'o')
    #    plt.plot(X_test[:, i:i + 1], final_labels, 'ro', markersize=3)
    #    plt.title((people.columns[i + 1] + ' -- lambda = ' + str(l / 10)))
    #    plt.show()

    #calculate f1_score for each one vs all fit, and add to list
    #the scores for each lambda value are separated by a zero f1
    for i in range(4):
        Y_test_labelled = select_label(Y_test, i)
        final_labels_labelled = select_label(final_labels, i)
        f1_scores.append(f1_score_(Y_test_labelled, final_labels_labelled))
        f1_scores.append(0)

f1_scores = [0 if i == None else i for i in f1_scores]
plt.bar(range(len(f1_scores)), f1_scores)
plt.show()
