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

    def setUp(self):
        edges = [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3)]
        self.g_undirected = GraphDataset.from_edges(n_nodes=5, edges=edges)
        self.g_directed = GraphDataset.from_edges(n_nodes=5, edges=edges, directed=True)

    def test_deg_list(self):
        self.assertListEqual(list(self.g_undirected.deg_list), [2, 4, 2, 1, 0])
        # Out degree
        self.assertListEqual(list(self.g_directed.deg_list), [2, 3, 0, 0, 0])

    def test_adj_list(self):
        self.assertSetEqual(set(self.g_undirected.adj_list[1].keys()), {0, 1, 2, 3})
        self.assertSetEqual(set(self.g_undirected.adj_list[2].keys()), {0, 1})
        self.assertSetEqual(set(self.g_directed.adj_list[0].keys()), {1, 2})
        self.assertEqual(len(self.g_directed.adj_list[2]), 0)

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

    def setUp(self):
        edges = [(0, 1, 2.0), (0, 2, 1.5), (1, 2, 0.4), (1, 3, 0.2)]
        self.g_weighted = GraphDataset.from_edges(n_nodes=4, edges=edges, weighted=True)
        self.g_unweighted = GraphDataset.from_edges(n_nodes=4, edges=edges)

    def test_unweighted(self):
        L = self.g_unweighted.get_laplacian()
        L_sparse = self.g_unweighted.get_laplacian(sparse=True).todense()
        L_expected = np.array([
            [ 2.0, -1.0, -1.0,  0.0],
            [-1.0,  3.0, -1.0, -1.0],
            [-1.0, -1.0,  2.0,  0.0],
            [ 0.0, -1.0,  0.0,  1.0]
        ])
        self.assertTrue(np.array_equal(L, L_expected))
        self.assertTrue(np.array_equal(L, L_sparse))

    def test_unweighted_normalized(self):
        Ln = self.g_unweighted.get_laplacian(normalize=True)
        Ln_sparse = self.g_unweighted.get_laplacian(normalize=True, sparse=True).todense()
        Ln_expected = np.array([
            [ 1.        , -0.40824829, -0.5       ,  0.        ],
            [-0.40824829,  1.        , -0.40824829, -0.57735027],
            [-0.5       , -0.40824829,  1.        ,  0.        ],
            [ 0.        , -0.57735027,  0.        ,  1.        ]
        ])
        self.assertTrue(np.allclose(Ln, Ln_expected))
        self.assertTrue(np.allclose(Ln, Ln_sparse))
    
    def test_weighted(self):
        L = self.g_weighted.get_laplacian()
        L_sparse = self.g_weighted.get_laplacian(sparse=True).todense()
        L_expected = np.array([
            [ 3.5, -2.0, -1.5,  0.0],
            [-2.0,  2.6, -0.4, -0.2],
            [-1.5, -0.4,  1.9,  0.0],
            [ 0.0, -0.2,  0.0,  0.2]
        ])
        self.assertTrue(np.array_equal(L, L_expected))
        self.assertTrue(np.array_equal(L, L_sparse))

class TestSubgraph(unittest.TestCase):

    def setUp(self):
        # A loop graph
        edges = [(0, 1, 0.1, 'a'), (1, 2, 0.2, 'b'), (2, 3, 0.3, 'c'),
                 (3, 4, 0.4, 'd'), (4, 5, 0.5, 'e'), (5, 0, 0.6, 'f')]
        node_labels = [11, 22, 33, 44, 55, 66]
        node_attributes = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        self.g_loop = GraphDataset.from_edges(n_nodes=6, edges=edges,
            weighted=True, directed=False, has_edge_data=True,
            node_labels=node_labels, node_attributes=node_attributes)

    def test_edge_removal(self):
        edges_to_remove = [(0, 1), (0, 2), (4, 3), (1, 0)]
        sg = self.g_loop.subgraph(edges_to_remove=edges_to_remove)
        expected_edges = [(1, 2, 0.2, 'b'), (2, 3, 0.3, 'c'),
                          (4, 5, 0.5, 'e'), (0, 5, 0.6, 'f')]
        self.assertSetEqual(set(sg.get_edge_iter(data=True)), set(expected_edges))
        self.assertListEqual(list(sg.deg_list), [1, 1, 2, 1, 1, 2])
    
    def test_node_removal(self):
        nodes_to_remove = [1, 4, 2, 3]
        sg = self.g_loop.subgraph(nodes_to_remove=nodes_to_remove)
        expected_edges = [(0, 1, 0.6, 'f')]
        self.assertSetEqual(set(sg.get_edge_iter(data=True)), set(expected_edges))
        expected_node_data = [(11, 'n1'), (66, 'n6')]
        self.assertSetEqual(set(zip(sg.node_labels, sg.node_attributes)), set(expected_node_data))
        self.assertListEqual(list(sg.deg_list), [1, 1])

    def test_node_keeping(self):
        nodes_to_keep = [2, 1]
        sg = self.g_loop.subgraph(nodes_to_keep=nodes_to_keep)
        expected_edges = [(0, 1, 0.2, 'b')]
        self.assertSetEqual(set(sg.get_edge_iter(data=True)), set(expected_edges))
        expected_node_data = [(22, 'n2'), (33, 'n3')]
        self.assertSetEqual(set(zip(sg.node_labels, sg.node_attributes)), set(expected_node_data))
        self.assertListEqual(list(sg.deg_list), [1, 1])

    def test_node_plus_edge_removal(self):
        nodes_to_remove = [0, 2]
        edges_to_remove = [(1, 0), (2, 3), (4, 5), (5, 0), (3, 4), (0, 1)]
        sg = self.g_loop.subgraph(nodes_to_remove=nodes_to_remove, edges_to_remove=edges_to_remove)
        self.assertEqual(sg.n_edges, 0)
        expected_node_data = [(22, 'n2'), (44, 'n4'), (55, 'n5'), (66, 'n6')]
        self.assertSetEqual(set(zip(sg.node_labels, sg.node_attributes)), set(expected_node_data))        
        self.assertListEqual(list(sg.deg_list), [0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()
