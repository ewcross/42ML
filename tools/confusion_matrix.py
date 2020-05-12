# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    confusion_matrix.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/10 14:15:19 by ecross            #+#    #+#              #
#    Updated: 2020/05/10 16:05:47 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd

def check_input(y, y_hat):
    if type(y) != np.ndarray or type(y_hat) != np.ndarray:
        return None, None, None
    if y.size == 0 or y_hat.size == 0:
        return None, None, None
    if y.ndim > 1:
        y = y.reshape((y.size,))
    if y_hat.ndim > 1:
        y_hat = y_hat.reshape((y_hat.size,))
    if y.shape != y_hat.shape:
        return None, None, None
    return 1, y, y_hat

def get_labels(y_true, y_hat, labels):
    if labels != None:
        return labels, dict.fromkeys(labels, 0)
    labels_true = set(y_true)
    labels_hat = set(y_hat)
    if len(labels_true) > len(labels_hat):
        labels = labels_true
    else:
        labels = labels_hat
    return sorted(labels), dict.fromkeys(labels, 0)

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    produces a confusion matrix given two numpy ndarrays, one
    containing the true classifications and one containing those predicted
    by a model - specific labels can be specified as a list, and the df
    option means the resulting matrix is returned as a DataFrame
    """
    check, y_true, y_hat = check_input(y_true, y_hat)
    if check is None:
        print('please check input arrays')
        return None
    labels, labels_dict = get_labels(y_true, y_hat, labels)
    for index, elem in enumerate(labels):
        new = [(x, y) for x, y in zip(y_true, y_hat) if x == elem]
        for tup in new:
            if tup[1] in labels_dict.keys():
                labels_dict[tup[1]] += 1
        if index == 0:
            matrix = np.array(list(labels_dict.values())).reshape(1, -1)
        else:
            lst = list(labels_dict.values())
            matrix = np.concatenate((matrix, np.array(lst).reshape(1, -1)), axis=0)
        for key in labels_dict:
            labels_dict[key] = 0
    if df_option == False:
        return matrix
    else:
        df = pd.DataFrame(matrix)
        df.columns = labels_dict.keys()
        df.index = labels
        return df

if __name__ == "__main__":

    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
    #print(confusion_matrix_(y_true, y_hat, ['dog', 'norminet']))
    print(confusion_matrix_(y_true, y_hat, df_option=True))
