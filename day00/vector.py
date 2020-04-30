# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vector.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:13:10 by ecross            #+#    #+#              #
#    Updated: 2020/04/30 11:48:09 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

class Vector:

    def __init__(self, var):
        if type(var) == list and self.check_float_list(var):
            i = 0
            for x in var:
                var[i] = float(var[i])
                i += 1
            self.vec = var
        elif type(var) == int or type(var) == float:
            self.vec = self.make_float_list(0, var)
        elif type(var) == tuple and len(var) == 2:
            self.vec = self.make_float_list(var[0], var[1])
        else:
            self.vec = []
        self.size = len(self.vec)

    def __repr__(self):
        return 'Vector: ' + str(self.vec)

    def __str__(self):
        return str(self.vec)

    def __add__(self, other):
        if not self.check_sum(other):
            return NotImplemented
        if isinstance(other, Vector):
            other = other.vec
        other = list(other)
        new_vec = []
        i = 0
        for x in self.vec:
            new_vec.append(self.vec[i] + float(other[i]))
            i += 1
        return Vector(new_vec)

    def __radd__(self, other):
        """carry out other + self, knowing that the class 'other'
            had no method for carrying out this operation itself"""
        if not self.check_sum(other):
            return NotImplemented
        other = list(other)
        return self.__add__(other)
    
    def __sub__(self, other):
        if not self.check_sum(other):
            return NotImplemented
        if isinstance(other, Vector):
            other = other.vec
        other = list(other)
        new_vec = []
        i = 0
        for x in self.vec:
            new_vec.append(self.vec[i] - float(other[i]))
            i += 1
        return Vector(new_vec)

    def __rsub__(self, other):
        """carry out other - self, knowing that the class 'other'
            had no method for carrying out this operation itself"""
        if not self.check_sum(other):
            return NotImplemented
        other = list(other)
        new_vec = []
        i = 0
        for x in self.vec:
            new_vec.append(float(other[i]) - self.vec[i])
            i += 1
        return Vector(new_vec)

    def __truediv__(self, other):
        check = self.check_prod(other)
        if check != 1:
            return NotImplemented
        new_vec = []
        other = float(other)
        i = 0
        for x in self.vec:
            new_vec.append(self.vec[i] / other)
            i += 1
        return Vector(new_vec)
    
    def __rtruediv__(self, other):
        """carry out other / self, knowing that the class 'other'
            had no method for carrying out this operation itself"""
        check = self.check_prod(other)
        if check != 1 and len(self.vec) != 1:
            return NotImplemented
        return float(other) / self.vec[0]

    def __mul__(self, other):
        check = self.check_prod(other)
        if not check:
            return NotImplemented
        if check == 1:
            new_vec = []
            other = float(other)
            i = 0
            for x in self.vec:
                new_vec.append(self.vec[i] * other)
                i += 1
            return Vector(new_vec)
        elif check == 2:
            res = 0
            if isinstance(other, Vector):
                other = other.vec
            i = 0
            for x in self.vec:
                res += self.vec[i] * float(other[i])
                i += 1
            return res
    
    def __rmul__(self, other):
        """carry out other * self, knowing that the class 'other'
            had no method for carrying out this operation itself"""
        return self.__mul__(other)

    def check_prod(self, other):
        """check that other is a scalar (int or float), or
            is a vector or list of same length as self.vec"""
        o_type = type(other)
        if o_type == int or o_type == float:
            return 1
        elif o_type == Vector:
            length = len(other.vec)
        elif o_type == list:
            length = len(other)
        else:
            return 0
        if not len(self.vec) == length:
            return 0
        return 2
    
    def check_float_list(self, var):
        for x in var:
            if not type(x) == float and not type(x) == int:
                return False
        return True

    def make_float_list(self, start, finish):
        v = []
        while start < finish:
            v.append(float(start))
            start += 1
        return v

    def check_sum(self, other):
        """check that the variable 'other' is a list or vector and
            that it has the same length as this vector"""
        o_type = type(other)
        if (o_type != list and o_type != Vector):
            return 0
        if o_type == Vector:
            length = len(other.vec)
        else:
            length = len(other)
        if not len(self.vec) == length:
            return 0
        return 1


v1 = Vector([1, 2, 3, 4, 5])
v2 = Vector([1, 2, 3, 4, 5])
print(v1 * 5)
