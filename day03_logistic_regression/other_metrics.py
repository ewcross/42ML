# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    other_metrics.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/10 10:29:02 by ecross            #+#    #+#              #
#    Updated: 2020/05/10 12:52:30 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def check_arrays(y, y_hat):
    if type(y) != np.ndarray or type(y_hat) != np.ndarray:
        return None, None
    y = y.reshape((-1,))
    y_hat = y_hat.reshape((-1,))
    if y.shape != y_hat.shape:
        return None, None
    return y, y_hat

def collect(y, y_hat, label):

    """gets number of true positives, false positives, true negatives
    and false negatives given a set of labels (1 or 0) and a corresponding
    predicted set - returns (tp, fp, tn, fn)"""
    
    y, y_hat = check_arrays(y, y_hat)
    if y is None:
        return None
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for tup in zip(y, y_hat):
        if tup[0] == label and tup[1] == label:
            tp += 1
        if tup[0] != label and tup[1] == label:
            fp += 1
        if tup[0] != label and tup[1] != label:
            tn += 1
        if tup[0] == label and tup[1] != label:
            fn += 1
    return tp, fp, tn, fn


def accuracy_score_(y, y_hat):

    """computes the accuracy score of a set of predicted labels,
    given the correct labels"""
    tp, fp, tn, fn = collect(y, y_hat, y.reshape(-1, 1)[0, 0])
    return (tp + tn) / y.shape[0]

def precision_score_(y, y_hat, pos_label=1):
    tp, fp, tn, fn = collect(y, y_hat, pos_label)
    if (tp + fp) == 0:
        return None
    return tp / (tp + fp)

def recall_score_(y, y_hat, pos_label=1):
    tp, fp, tn, fn = collect(y, y_hat, pos_label)
    return tp / (tp + fn)

def f1_score_(y, y_hat, pos_label=1):
    tp, fp, tn, fn = collect(y, y_hat, pos_label)
    precision = precision_score_(y, y_hat, pos_label)
    if precision is None:
        return None
    recall = recall_score_(y, y_hat, pos_label)
    return (2 * precision * recall) / (precision + recall)

if __name__ == "__main__":

    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print(accuracy_score_(y, y_hat))
    print(accuracy_score(y, y_hat))
    print()
    print(precision_score_(y, y_hat, 'dog'))
    print(precision_score(y, y_hat, pos_label='dog'))
    print()
    print(recall_score_(y, y_hat, 'dog'))
    print(recall_score(y, y_hat, pos_label='dog'))
    print()
    print(f1_score_(y, y_hat, 'dog'))
    print(f1_score(y, y_hat, pos_label='dog'))
    print()
    print()
    print()
    print(accuracy_score_(y, y_hat))
    print(accuracy_score(y, y_hat))
    print()
    print(precision_score_(y, y_hat, 'norminet'))
    print(precision_score(y, y_hat, pos_label='norminet'))
    print()
    print(recall_score_(y, y_hat, 'norminet'))
    print(recall_score(y, y_hat, pos_label='norminet'))
    print()
    print(f1_score_(y, y_hat, 'norminet'))
    print(f1_score(y, y_hat, pos_label='norminet'))
