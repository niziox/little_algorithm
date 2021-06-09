#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np



def all_get_vertex_cost(G):
    zeros_tab = []

    for row in range(G.shape[0]):
        for col in range(G.shape[1]):
            if G[row, col] == 0:
                get_vertex_cost(G, row, col, zeros_tab)








if __name__ == '__main__':
    cost_matrix = np.array([[np.inf, 10, 8, 19, 12],
                      [10, np.inf, 20,  6,  3],
                      [8,   20, np.inf, 4,  2],
                      [19,  6,  4, np.inf,  7],
                      [12,  3,  2,   7, np.inf]])

    # print(get_vertex_cost(reduction(cost_matrix)[0], 0, 2))

class CostMatrix:
    def __init__(self, m):
        self.matrix = m

    def size(self):
        return len(self.matrix)

    def reduce_rows(self):
        # odejęcie minimalnej wartości w każdym wierszu
        row = self.matrix.min(axis=1)
        self.matrix = self.matrix - np.array([row]).T
        return sum(row)

    def reduce_cols(self):
        # odejęcie minimalnej wartości w każdej kolumnie
        col = self.matrix.min(axis=0)
        self.matrix = self.matrix - col

        return sum(col)

    def get_vertex_cost(self, row, col):
        min_rows = self.matrix[row, :]
        min_cols = self.matrix[:, col]

        min_row = np.inf
        min_col = np.inf

        for i in range(len(min_rows)):
            if i != col and min_rows[i] < min_row:
                min_row = min_rows[i]

            if i != row and min_cols[i] < min_col:
                min_col = min_cols[i]

        return min_row + min_col

class VertexT:
    def __init__(self, row=0, col=0):
        self.row = row
        self.col = col

class NewVertex:
    def __init__(self, v=(0, 0), cost=0):
        self.coordinates = VertexT(v)
        self.cost = cost

class StageState:
    def __init__(self, m, p, lb):
        self.matrix_ = m
        self.unsorted_path = p
        self.lower_bound_ = lb

    def get_path(self):
        sorted_path = []
        legit_vertices = []
        for i in range(len(self.matrix_)):
            for j in range(len(self.matrix_)):
                if self.matrix_[i, j] != np.inf:
                    legit_vertices.append(VertexT(i, j))
        first_p = []
        second_p = []

        for elem in self.unsorted_path:
            first_p.append(elem.row)
            second_p.append(elem.col)

        unsorted_copy = self.unsorted_path.copy()
        for new_v in legit_vertices:
            check_1 = True
            check_2 = True
            for i in range(len(first_p)):
                if new_v.col == first_p[i]:
                    check_1 = False
                if new_v.row ==second_p[i]:
                    check_2 = False
            if check_1 or check_2:
                self.append_to_path(new_v)

        sorted_path.append(self.unsorted_path[0].row)
        next = self.unsorted_path[0].col
        run = True
        while run:
            for i in range(1, len(self.unsorted_path)):
                if self.unsorted_path[i] == next:
                    sorted_path.append(self.unsorted_path[i].row)
                    next = self.unsorted_path[i].col
                    break
                if i+1 == len(self.unsorted_path):
                    run = False

        return sorted_path

    def append_to_path(self, v):
        self.unsorted_path.append(v)

    def get_level(self):
        return len(self.unsorted_path)

    def update_lower_bound(self, reduced_values):
        self.lower_bound_ += reduced_values

    def get_lower_bound(self):
        return self.lower_bound_

    def reset_lower_bound(self):
        self.lower_bound_ = 0

    def reduce_cost_matrix(self):
        sum = 0
        sum += self.matrix_.reduce_rows() + self.matrix_.reduce_cols()

    def choose_new_vertex(self):
        vertex_list = []
        for row in range(len(self.matrix_)):
            for col in range(len(self.matrix_[row])):
                if self.matrix_[row, col] == 0:
                    temp = {}
                    temp[(row, col)] = self.matrix_.get_vertex_cost(row, col)
                    vertex_list.append(temp)

        coords = list(vertex_list[0].keys())[0]
        cost = vertex_list[0][coords]
        for pair in vertex_list:
            if list(pair.values())[0] > cost and list(pair.values())[0] != np.inf:
                coords = list(pair.keys())[0]
                cost = list(pair.values())[0]
        coords = VertexT(coords[0], coords[1])
        return NewVertex()



    def update_cost_matrix(self, new_vertex):
        self.matrix_[new_vertex.col, new_vertex.row] = np.inf
        for elem in self.matrix_[new_vertex.row]:
            elem = np.inf

        for i in range(len(self.matrix_)):
            self.matrix_[i, new_vertex.col] = np.inf