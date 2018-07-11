import numpy as np
import warnings

def laplacian_eigenmaps(graph, n_components=2):
    """
    Computes the Lapacian eigenmap.

    Ref: M. Belkin, P. Niyogi, Laplacian eigenmaps and spectral techniques for
         embedding and clustering, in: NIPS, Vol. 14, 2001, pp. 585-591.
    """
    if graph.n_nodes > 10000:
        warnings.warn('The default implementation computes the full eigendecomposition, which is not efficient for large graphs.')
    if graph.directed:
        raise ValueError('Graph should be undirected.')
    if n_components < 0 or n_components > graph.n_nodes:
        raise ValueError('Number of components must be positive and less than the number of nodes.')
    L = graph.get_laplacian(normalize=True)
    l, v = np.linalg.eig(L)
    ind = np.argsort(np.abs(l))
    return v[:, ind[1:1+n_components]]

