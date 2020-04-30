# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    TinyStatistician.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ecross <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 13:02:41 by ecross            #+#    #+#              #
#    Updated: 2020/04/30 15:40:20 by ecross           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math

class TinyStatistician:

    def mean(self, x):
        l = len(x)
        if not l:
            return None
        mean = 0
        for val in x:
            mean += val
        return mean / l

    def median(self, x, pc=50):
        pc = pc * 0.01
        l = len(x)
        if not l:
            return None
        x.sort()
        index = l * pc
        if l % (1 / pc):
            return x[int(index + 1) - 1]
        else:
            return (x[index - 1] + x[index]) / 2

    def quartiles(self, x, percentile):
        return self.median(x, percentile)

    def var(self, x):
        l = len(x)
        if not l:
            return None
        mean = self.mean(x)
        var =0
        for val in x:
            var += (val - mean) * (val - mean)
        return var / l

    def std(self, x):
        return math.sqrt(self.var(x))

if __name__ == "__main__":

    tstat = TinyStatistician()
    a = [1, 42, 300, 10, 59]
    a.sort()
    print(a)
    print(tstat.mean(a))
    print(tstat.median(a))
    print(tstat.quartiles(a, 25))
    print(tstat.quartiles(a, 75))
    print(tstat.var(a))
    print(tstat.std(a))
