import unittest
import numpy as np
from graphz.dataset import GraphDataset

class TestCreation(unittest.TestCase):

    def test_from_simple_edge_list(self):
        edges = [(0, 0), (0, 1), (0, 2), (0, 3), (3, 4)]
        g = GraphDataset.from_edges(n_nodes=5, edges=edges, name='GraphX')
        self.assertEqual(g.name, 'GraphX')
        self.assertFalse(g.weighted)
        self.assertFalse(g.directed)
        self.assertEqual(g.n_nodes, 5)
        self.assertEqual(g.n_edges, 5)
        self.assertSetEqual(set(g.get_edge_iter()), set(map(lambda e : (e[0], e[1], 1), edges)))

class TestViews(unittest.TestCase):

    def test_deg_list(self):
        edges = [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3)]
        g = GraphDataset.from_edges(n_nodes=5, edges=edges)
        self.assertListEqual(list(g.deg_list), [2, 4, 2, 1, 0])

class TestAdjacencyMatrix(unittest.TestCase):

    def test_undirected(self):
        edges = [(0, 1, 0.1), (1, 2, 0.2), (2, 3, 0.3)]
        g = GraphDataset.from_edges(n_nodes=4, edges=edges, weighted=True)
        A = g.get_adj_matrix()
        A_expected = np.array([
            [0.0, 0.1, 0.0, 0.0],
            [0.1, 0.0, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.3],
            [0.0, 0.0, 0.3, 0.0]
        ])
        self.assertTrue(np.array_equal(A, A_expected))

    def test_directed(self):
        edges = [(0, 1, 0.1), (1, 2, 0.2), (3, 3, 0.5)]
        g = GraphDataset.from_edges(n_nodes=4, edges=edges, weighted=True, directed=True)
        A = g.get_adj_matrix()
        A_expected = np.array([
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5]
        ])
        self.assertTrue(np.array_equal(A, A_expected))

class TestLaplacianMatrix(unittest.TestCase):

    def test_unweighted(self):
        edges = [(0, 1), (0, 2), (1, 2), (1, 3)]
        g = GraphDataset.from_edges(n_nodes=4, edges=edges)
        L = g.get_laplacian()
        L_sparse = g.get_laplacian(sparse=True).todense()
        L_expected = np.array([
            [ 2.0, -1.0, -1.0,  0.0],
            [-1.0,  3.0, -1.0, -1.0],
            [-1.0, -1.0,  2.0,  0.0],
            [ 0.0, -1.0,  0.0,  1.0]
        ])
        self.assertTrue(np.array_equal(L, L_expected))
        self.assertTrue(np.array_equal(L, L_sparse))


if __name__ == '__main__':
    unittest.main()
