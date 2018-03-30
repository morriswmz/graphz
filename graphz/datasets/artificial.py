from graphz.dataset import GraphDataset
import numpy as np
import random

def create_loop_graph(n = 5):
    """
    Creates a simple loop with n nodes.
    """
    if n < 0 or int(n) != n:
        raise ValueError('n must be a nonnegative integer.')
    edges = map(lambda i : (i, (i + 1) % n), range(n))
    return GraphDataset.from_edges(n_nodes=n, edges=edges, weighted=False,
                                   directed=False, name="Loop-{}".format(n))
        
def create_random_graph(n = 10, p = 0.1):
    """
    Creates a simple random graph, where each edge is added with
    probability p.
    """
    if n < 0 or int(n) != n:
        raise ValueError('n must be a nonnegative integer.')
    if p < 0. or p > 1.:
        raise ValueError('p must be a probability.')
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() <= p:
                edges.append((i, j))
    return GraphDataset.from_edges(n_nodes=n, edges=edges, weighted=False,
                                   directed=False, name="RandomGraph-{}-{}".format(n, p))
