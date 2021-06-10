import unittest
import numpy as np
from little_algorithm import CostMatrix, StageState, VertexT, NewVertex


class TestCostMatrix(unittest.TestCase):
    def test_reduce_rows(self):
        cm = np.array([
            [np.inf, 12, 3, 45, 6],
            [78, np.inf, 90, 21, 3],
            [5, 56, np.inf, 23, 98],
            [12, 6, 8, np.inf, 34],
            [3, 98, 3, 2, np.inf]
        ])

        test_matrix = CostMatrix(cm)

        expected_m_t = np.array([
            [np.inf, 9, 0, 42, 3],
            [75, np.inf, 87, 18, 0],
            [0, 51, np.inf, 18, 93],
            [6, 0, 2, np.inf, 28],
            [1, 96, 1, 0, np.inf]
        ])

        expected_m = CostMatrix(expected_m_t)
        expected_reduction = 19

        self.assertEqual(expected_reduction, test_matrix.reduce_rows())
        self.assertEqual(True, np.all(expected_m.matrix == test_matrix.matrix))
        
    def test_reduce_cols(self):
        cm = np.array([
            [np.inf, 12, 3, 45, 6],
            [78, np.inf, 90, 21, 3],
            [5, 56, np.inf, 23, 98],
            [12, 6, 8, np.inf, 34],
            [3, 98, 3, 2, np.inf]
        ])

        test_matrix = CostMatrix(cm)

        expected_m_t = np.array([
            [np.inf, 6, 0, 43, 3],
            [75, np.inf, 87, 19, 0],
            [2, 50, np.inf, 21, 95],
            [9, 0, 5, np.inf, 31],
            [0, 92, 0, 0, np.inf]
        ])

        expected_m = CostMatrix(expected_m_t)
        expected_reduction = 17

        self.assertEqual(expected_reduction, test_matrix.reduce_cols())
        self.assertEqual(True, np.all(expected_m.matrix == test_matrix.matrix))

    def test_get_vertex_cost(self):
        sample_m = np.array([
            [np.inf, 1, 0, 9, 4],
            [1, np.inf, 17, 1, 0],
            [0, 17, np.inf, 0, 0],
            [9, 1, 0, np.inf, 3],
            [4, 0, 0, 3, np.inf]
        ])
        sample_matrix = CostMatrix(sample_m)

        self.assertEqual(1, sample_matrix.get_vertex_cost(row=0, col=2))
        self.assertEqual(1, sample_matrix.get_vertex_cost(row=2, col=0))
        self.assertEqual(0, sample_matrix.get_vertex_cost(row=2, col=4))
        self.assertEqual(1, sample_matrix.get_vertex_cost(row=4, col=1))
        self.assertEqual(1, sample_matrix.get_vertex_cost(row=2, col=3))
        self.assertEqual(1, sample_matrix.get_vertex_cost(row=3, col=2))
        self.assertEqual(0, sample_matrix.get_vertex_cost(row=4, col=2))


class TestStageState(unittest.TestCase):
    def test_reduce_cost_matrix(self):
        sample_m = np.array([
            [np.inf, 10, 8, 19, 12],
            [10, np.inf, 20, 6, 3],
            [8, 20, np.inf, 4, 2],
            [19, 6, 4, np.inf, 7],
            [12, 3, 2, 7, np.inf]
        ])

        sample_matrix = CostMatrix(sample_m)

        ans_sample_m = np.array([
            [np.inf, 1, 0, 9, 4],
            [1, np.inf, 17, 1, 0],
            [0, 17, np.inf, 0, 0],
            [9, 1, 0, np.inf, 3],
            [4, 0, 0, 3, np.inf]
        ])
        sample_stage_state = StageState(m=sample_matrix)
        expected_reduction_cost = 28

        self.assertEqual(expected_reduction_cost, sample_stage_state.reduce_cost_matrix())
        self.assertEqual(True, np.all(ans_sample_m == sample_stage_state.matrix_.matrix))

    def test_get_path(self):
        sample_m = np.array([
            [np.inf, np.inf, np.inf, np.inf, np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
            [np.inf, 0, np.inf, np.inf, 2],
            [np.inf, 0, np.inf, np.inf, np.inf]
        ])
        sample_matrix = CostMatrix(sample_m)
        sample_unsorted_path = [VertexT(0, 2), VertexT(1, 0), VertexT(2, 3)]
        sample_stage_state = StageState(m=sample_matrix, p=sample_unsorted_path, lb=30)
        expected_path = [2, 3, 4, 1, 0]
        self.assertEqual(expected_path, sample_stage_state.get_path())

    def test_choose_new_vertex(self):
        cm = np.array([
            [np.inf, 2, 1, 11, 4],
            [7, np.inf, 17, 3, 0],
            [5, 18, np.inf, 2, 0],
            [15, 2, 0, np.inf, 3],
            [10, 1, 1, 5, np.inf]
        ])
        t = StageState(CostMatrix(cm))
        v1 = NewVertex(VertexT(1, 4), 3)
        v2 = t.choose_new_vertex()

        self.assertEqual(v1.cost, v2.cost)
        self.assertEqual(v1.coordinates.row, v2.coordinates.row)
        self.assertEqual(v1.coordinates.col, v2.coordinates.col)
        self.assertEqual(v1.cost, t.matrix_.get_vertex_cost(1, 4))


if __name__ == '__main__':
    unittest.main()
