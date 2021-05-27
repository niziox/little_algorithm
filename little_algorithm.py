#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def reduction(G):
    # odejęcie minimalnej wartości w każdym wierszu
    row = G.min(axis=1)
    G = G - np.array([row]).T

    # odejęcie minimalnej wartości w każdej kolumnie
    col = G.min(axis=0)
    G = G - col

    return G, sum(row) + sum(col)



if __name__ == '__main__':
    pass