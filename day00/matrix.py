# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    matrix.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 11:53:00 by ecross            #+#    #+#              #
#    Updated: 2020/04/30 12:59:27 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

class Matrix:

    def __init__(self, *args):
        l = len(args)
        if l == 0 or l > 2:
            raise ValueError
        elif l == 1:
            if isinstance(args[0], list):
                self.init_list(args[0])
            elif isinstance(args[0], tuple):
                self.init_shape(args[0])
            else:
                raise ValueError
        elif l == 2:
            if not (isinstance(args[0], list) and isinstance(args[1], tuple)):
                raise ValueError
            self.init_list(args[0], args[1])

    def __repr__(self):
        return "Matrix: " + str(self.data)

    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if not (self.shape == other.shape):
            return NotImplemented
        new_matrix = []
        i = 0
        for rself in self.data:
            new_row = []
            j = 0
            for cself in rself:
                new_row.append(cself + other.data[i][j])
                j += 1
            new_matrix.append(new_row)
            i += 1
        return Matrix(new_matrix)
    
    def init_list(self, lists, shape=None):
        if shape:
            if not len(lists) == shape[0]:
                raise ValueError
        for l in lists:
            if not isinstance(l, list):
                raise ValueError
            for elem in l:
                if not isinstance(elem, float) and not isinstance(elem, int):
                    raise ValueError
        self.data = []
        if shape:
            self.shape = shape
            length = shape[1]
        else:
            length = len(lists[0])
        for l in lists:
            if not len(l) == length:
                raise ValueError
            i = 0
            for elem in l:
                l[i] = (float(elem))
                i += 1
            self.data.append(l)
            self.shape = (int(len(lists)), int(length))

    def init_shape(self, shape):
        self.shape = shape
        self.data = []
        row = [0] * shape[1]
        for i in range(shape[0]):
            self.data.append(row)



if __name__ == "__main__":

    matrix1 = Matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    matrix2 = Matrix([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    print(matrix1 + matrix2)
