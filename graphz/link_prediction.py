import numpy as np
import math

def common_neighbors(graph, pairs):
    r"""
    Computes the number of common neighbors for the given node pairs:
        |N(x) \cap N(y)|

    :param graph: Graph input.
    :param pairs: List of node pair tuples.
    
    Note: the implementation based on undirected graphs. For directed graphs
          only common out neighbors are considered.
    """
    cn = np.zeros(len(pairs))
    adj_list = graph.adj_list
    for i, e in enumerate(pairs):
        sa = set(adj_list[e[0]].keys())
        sb = set(adj_list[e[1]].keys())
        cn[i] = len(sa & sb)
    return cn

def jaccard_coeff(graph, pairs):
    r"""
    Computes the Jaccard's coefficients for the given node pairs:
        |N(x) \cap N(y)| / |N(x) \cup N(y)|

    :param graph: Graph input.
    :param pairs: List of node pair tuples.

    Note: the implementation based on undirected graphs. For directed graphs
          only common out neighbors are considered.
    """
    coeff = np.zeros(len(pairs))
    adj_list = graph.adj_list
    for i, e in enumerate(pairs):
        sa = set(adj_list[e[0]].keys())
        sb = set(adj_list[e[1]].keys())
        if len(sa) == 0 and len(sb) == 0:
            # If both sets are empty, define it as one.
            coeff[i] = 1.
        else:
            ni = len(sa & sb)
            coeff[i] = ni / (len(sa) + len(sb) - ni)
    return coeff

def adamic_adar(graph, pairs):
    r"""
    Computes the Adamic/Adar similarity measure for the given node pairs:
        \sum_{z \in N(x) \cap N(y)} 1/log(N(z))

    :param graph: Graph input.
    :param pairs: List of node pair tuples.

    Note: the implementation based on undirected graphs. For directed graphs
          only common out neighbors are considered.
    """
    coeff = np.zeros(len(pairs))
    adj_list = graph.adj_list
    deg_list = graph.deg_list
    for i, e in enumerate(pairs):
        sc = set(adj_list[e[0]].keys()) & set(adj_list[e[1]].keys())
        coeff[i] = sum([1.0 / math.log(deg_list[i]) for i in sc if deg_list[i] >= 2])
    return coeff

def preferential_attachment(graph, pairs):
    r"""
    Computes the preferential attachment measure for the given node pairs:
        |N(x)| |N(y)|

    Note: the implementation based on undirected graphs. For directed graphs
          only common out neighbors are considered.
    """
    pa = np.zeros(len(pairs))
    deg_list = graph.deg_list
    for i, e in enumerate(pairs):
        pa[i] = deg_list[e[0]] * deg_list[e[1]]
    return pa

def resistance_distance(graph, pairs):
    """
    Computes resistance distance.
    See http://mathworld.wolfram.com/ResistanceDistance.html.

    Note: this procedure requires computing the pseudo inverse of graph
          Laplacian, which is time consuming for large graphs.
    """
    l_inv = np.linalg.pinv(graph.get_laplacian())
    l_inv_diag = np.diag(l_inv).reshape((-1, 1))
    omega = l_inv_diag + l_inv_diag.T - l_inv - l_inv.T
    rd = np.zeros(len(pairs))
    for i, e in enumerate(pairs):
        rd[i] = omega[e[0], e[1]]
    return rd

def katz(graph, pairs, beta=0.001):
    r"""
    Computes Katz measure: (I - \beta A)^{-1} - I
    Reference:
    * Leo Katz. A new status index derived from sociometric analysis.
      Psychometrika, 18(1):39–43, March 1953.
    """
    A = graph.get_adj_matrix()
    n = A.shape[0]
    M = np.inv(np.eye(n) - beta * A) - np.eye(n)
    km = np.zeros(len(pairs))
    for i, e in enumerate(pairs):
        km[i] = M[e[0], e[1]]
    return km