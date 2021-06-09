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

def get_vertex_cost(G, row, col):
    min_rows = G[row, :]
    min_cols = G[:, col]

    min_row = np.inf
    min_col = np.inf

    for i in range(len(min_rows)):
        if i != col and min_rows[i] < min_row:
            min_row = min_rows[i]

        if i != row and min_cols[i] < min_col:
            min_col = min_cols[i]

    return min_row + min_col


if __name__ == '__main__':
    cost_matrix = np.array([[np.inf, 10, 8, 19, 12],
                      [10, np.inf, 20,  6,  3],
                      [8,   20, np.inf, 4,  2],
                      [19,  6,  4, np.inf,  7],
                      [12,  3,  2,   7, np.inf]])

    print(get_vertex_cost(reduction(cost_matrix)[0], 0, 2))
