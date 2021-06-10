#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from typing import List


class CostMatrix:
    def __init__(self, m):
        self.matrix = m

    def size(self):
        return len(self.matrix)

    def reduce_rows(self):
        # odjęcie minimalnej wartości w każdym wierszu
        row = self.matrix.min(axis=1)
        self.matrix = self.matrix - np.array([row]).T
        return sum(row)

    def reduce_cols(self):
        # odjęcie minimalnej wartości w każdej kolumnie
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
        if isinstance(v, VertexT):
            self.coordinates = VertexT(v.row, v.col)
        else:
            self.coordinates = VertexT(*v)
        self.cost = cost


class StageState:
    def __init__(self, m, p=None, lb=0):
        if p is None:
            p = []
        self.matrix_ = m
        self.unsorted_path = p
        self.lower_bound_ = lb

    def get_path(self):
        #self.reduce_cost_matrix()
        sorted_path = []
        legit_vertices = []
        for i in range(self.matrix_.size()):
            for j in range(self.matrix_.size()):
                if self.matrix_.matrix[i, j] != np.inf:
                    legit_vertices.append(VertexT(i, j))
        first_p = []
        second_p = []

        for elem in self.unsorted_path:
            first_p.append(elem.row)
            second_p.append(elem.col)

        #unsorted_copy = self.unsorted_path.copy()
        for new_v in legit_vertices:
            check_1 = True
            check_2 = True
            for i in range(len(first_p)):
                if new_v.col == first_p[i]:
                    check_1 = False
                if new_v.row == second_p[i]:
                    check_2 = False
            if check_1 or check_2:
                self.append_to_path(new_v)

        sorted_path.append(self.unsorted_path[0].row)
        print(len(self.unsorted_path))
        next = self.unsorted_path[0].col
        run = True
        while run:
            for i in range(1, len(self.unsorted_path)):
                if self.unsorted_path[i].row == next:
                    sorted_path.append(self.unsorted_path[i].row)
                    next = self.unsorted_path[i].col
                    break
                print(i, len(self.unsorted_path))
                if i + 1 == len(self.unsorted_path):
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
        return self.matrix_.reduce_rows() + self.matrix_.reduce_cols()

    def choose_new_vertex(self):
        vertex_list = []
        for row in range(self.matrix_.size()):
            for col in range(self.matrix_.size()):
                if self.matrix_.matrix[row, col] == 0:
                    temp = {(row, col): self.matrix_.get_vertex_cost(row, col)}
                    vertex_list.append(temp)
        coords = list(vertex_list[0].keys())[0]
        cost = vertex_list[0][coords]
        for pair in vertex_list:
            if list(pair.values())[0] > cost and list(pair.values())[0] != np.inf:
                coords = list(pair.keys())[0]
                cost = list(pair.values())[0]
        coords = VertexT(coords[0], coords[1])
        return NewVertex(coords, cost)

    def update_cost_matrix(self, new_vertex):
        self.matrix_.matrix[new_vertex.col, new_vertex.row] = np.inf
        for i in range(self.matrix_.size()):
            self.matrix_.matrix[new_vertex.row, i] = np.inf

        for i in range(self.matrix_.size()):
            self.matrix_.matrix[i, new_vertex.col] = np.inf


class TSPSolution:
    def __init__(self, lower_bound, path):
        self.lower_bound = lower_bound
        self.path = path


cost_t = int


def get_optimal_cost(optimal_path, m) -> cost_t:
    cost = 0
    # zsumowanie wag krawędzi w ścieżce optymalnej
    for idx in range(1, len(optimal_path)):
        cost += m.matrix[optimal_path[idx - 1]][optimal_path[idx]]
    # dodanie kosztu powrotu do wierzchołka początkowego
    cost += m.matrix[optimal_path[len(optimal_path) - 1]][optimal_path[0]]
    return cost


def create_right_branch_matrix(m, vertex, lower_bound) -> StageState:
    m.matrix[vertex.row, vertex.col] = np.inf
    return StageState(m=m, p=[], lb=lower_bound)


def filter_solutions(solutions: List[TSPSolution]) -> List[TSPSolution]:
    optimal_cost = min(solutions, key=lambda s: s.lower_bound).lower_bound
    optimal_solutions = list(filter(lambda s: s.lower_bound == optimal_cost, solutions))
    return optimal_solutions


def solve_tsp(cm):

    left_branch = StageState(cm)
    tree_lifo = [left_branch]
    n_levels = cm.size() - 2

    best_lb = np.inf
    solutions = []

    while tree_lifo:
        left_branch = tree_lifo.pop()

        while left_branch.get_level() != n_levels and left_branch.get_lower_bound() <= best_lb:

            if left_branch.get_level() == 0:
                left_branch.reset_lower_bound()

            new_cost = left_branch.reduce_cost_matrix()
            left_branch.update_lower_bound(new_cost)
            if left_branch.get_lower_bound() > best_lb:
                break

            new_vertex = left_branch.choose_new_vertex()

            left_branch.append_to_path(new_vertex.coordinates)
            left_branch.update_cost_matrix(new_vertex.coordinates)
            new_lower_bound = left_branch.get_lower_bound() + new_vertex.cost
            tree_lifo.append(create_right_branch_matrix(cm, new_vertex.coordinates, new_lower_bound))

        if left_branch.get_lower_bound() <= best_lb:
            best_lb = left_branch.get_lower_bound()
            new_path = left_branch.get_path()
            solutions.append(TSPSolution(get_optimal_cost(new_path, cm), new_path))

    return filter_solutions(solutions)


if __name__ == '__main__':
    cost_matrix = np.array([[np.inf, 2, 80, 95, 76, 78],
                            [2, np.inf, 3, 60, 70, 67],
                            [80, 3, np.inf, 4, 69, 80],
                            [95, 60, 4, np.inf, 5, 34],
                            [76, 70, 69, 5, np.inf, 6],
                            [78, 67, 80, 34, 6, np.inf]])
    solutions = solve_tsp(CostMatrix(cost_matrix))
    for s in solutions:
        print(s.path)
    print('done')
