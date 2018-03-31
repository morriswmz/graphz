import numpy as np
import copy
from collections import namedtuple
from graphz.view import DictionaryView, ListView, AdjacencyListView
try:
    import networkx
    networkx_installed = True
except ImportError:
    networkx_installed = False

"""
Stores information of an edge in adjacency lists.
"""
EdgeInfo = namedtuple('EdgeInfo', ['weight', 'attributes'])

"""
Provides full information of a node.
"""
NodeView = namedtuple('NodeView', ['id', 'out_edges', 'label', 'attributes'])

# Represents an edge with unit weight and no attributes.
UW_NA_EDGE_INFO = EdgeInfo(1, None) 

def check_node_index(nid, n_nodes):
    if nid < 0 or nid >= n_nodes:
        raise ValueError('Node id {} is out of bounds.'.format(nid))

def merge_edge(na, nb, eia, eib):
    pass

def build_adj_list_undirected(n_nodes, edges, weighted=False, attributes=False):
    """
    Builds a adjacency list from an edge list for an undirected graph.

    :param n_nodes: Number of nodes.
    
    :param edges: Collection of edges. Each item must have the following
    structure: (na, nb[, w, attr]). When weighted is True, w must be present.
    When attributes is True, both w and attr must be present (even if w will
    be ignored if weighted is False). Examples:
    1. weighted=False, attributes=False
        edges = [(0, 1), (0, 2), (1, 2)]
    2. weighted=True, attributes=False
        edges = [(0, 1, 0.2), (0, 2, 0.4), (1, 2, 0.3)]
    3. weighted=False, attributes=True
        edges = [(0, 1, 1, 'e1'), (0, 2, 1, 'e2'), (1, 2, 1, 'e3')]
    4. weighted=True, attributes=True
        edges = [(0, 1, 0.2, 'e1'), (0, 2, 0.4, 'e2'), (1, 2, 0.3, 'e3')]
    Note: when attributes is True, edges cannot contain duplicate definitions.
    
    :param weighted: Specifies whether the edges have weights.

    :param attributes: Specified whether the edges have attributed.

    :return: The constructed adjacency list.
    """
    adj_list = [{} for i in range(n_nodes)]
    if weighted:
        if attributes:
            # Attributes provided.
            # If an edge pair appear multiple times, an error will be raised
            # because we do not know how to merge the attributes.
            for e in edges:
                na, nb, w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                # For undirected graphs, adj_list[na][nb] and adj_list[nb][na]
                # are updated simultaneously. We only need to check one of them.
                if nb in adj_list[na]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when attributes are provided.'.format(na, nb))
                ei = EdgeInfo(w, attr)
                adj_list[na][nb] = ei
                adj_list[nb][na] = ei
        else:
            # No attributes provided.
            # If an edge pair appears multiple times, we combine them by summing
            # their weights.
            for e in edges:
                na, nb, w = e[0], e[1], e[2]
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if nb in adj_list[na]:
                    # Merge weights.
                    adj_list[na][nb] = EdgeInfo(w + adj_list[na][nb].weight, None)
                else:
                    adj_list[na][nb] = EdgeInfo(w, None)
                adj_list[nb][na] = adj_list[na][nb]
    else:
        if attributes:
            for e in edges:
                na, nb, _w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if nb in adj_list[na]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when attributes are provided.'.format(na, nb))
                ei = EdgeInfo(1, attr)
                adj_list[na][nb] = ei
                adj_list[nb][na] = ei
        else:
            # When no node attributes are present, duplications are ignored for
            # undirected graphs.
            for e in edges:
                na, nb = e[0], e[1]
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                adj_list[na][nb] = UW_NA_EDGE_INFO
                adj_list[nb][na] = UW_NA_EDGE_INFO
    return adj_list

def build_adj_list_directed(n_nodes, edges, weighted=False, attributes=False):
    """
    Builds a adjacency list from an edge list for a directed graph.

    :param n_nodes: Number of nodes.
    
    :param edges: Collection of edges. Each item must have the following
    structure: (na, nb[, w, attr]). When weighted is True, w must be present.
    When attributes is True, both w and attr must be present (even if w will
    be ignored if weighted is False). Examples:
    1. weighted=False, attributes=False
        edges = [(0, 1), (0, 2), (1, 2)]
    2. weighted=True, attributes=False
        edges = [(0, 1, 0.2), (0, 2, 0.4), (1, 2, 0.3)]
    3. weighted=False, attributes=True
        edges = [(0, 1, 1, 'e1'), (0, 2, 1, 'e2'), (1, 2, 1, 'e3')]
    4. weighted=True, attributes=True
        edges = [(0, 1, 0.2, 'e1'), (0, 2, 0.4, 'e2'), (1, 2, 0.3, 'e3')]
    Note: when attributes is True, edges cannot contain duplicate definitions.
    
    :param weighted: Specifies whether the edges have weights.

    :param attributes: Specified whether the edges have attributed.

    :return: The constructed adjacency list.
    """
    adj_list = [{} for i in range(n_nodes)]
    if weighted:
        if attributes:
            # Attributes provided.
            # If an edge pair appear multiple times, an error will be raised
            # because we do not know how to merge the attributes.
            for e in edges:
                na, nb, w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if nb in adj_list[na]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when attributes are provided.'.format(na, nb))
                adj_list[na][nb] = EdgeInfo(w, attr)
        else:
            # No attributes provided.
            # If an edge pair appears multiple times, we combine them by summing
            # their weights.
            for e in edges:
                na, nb, w = e[0], e[1], e[2]
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if nb in adj_list[na]:
                    # Merge weights.
                    adj_list[na][nb] = EdgeInfo(w + adj_list[na][nb].weight, None)
                else:
                    adj_list[na][nb] = EdgeInfo(w, None)
    else:
        if attributes:
            for e in edges:
                na, nb, _w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if na in adj_list[nb]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when attributes are provided.'.format(na, nb))
                adj_list[na][nb] = EdgeInfo(1, attr)
        else:
            # When no node attributes are present, duplications are ignored for
            # undirected graphs.
            for e in edges:
                na, nb = e[0], e[1]
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                adj_list[na][nb] = UW_NA_EDGE_INFO
    return adj_list

def enumerate_edges_undirected(adj_list, attributes=False):
    """
    Creates a generator that iterative through all the edges based on the give
    adjacency list, assuming an undirected graph. Each item will be a tuple
    of the format: (na, nb, w[, attr]).

    :param adj_list: Adjacency list.
    
    :param attributes: If set to true, edge attributes will be included.
    """
    # For undirected graphs, we do not want to record both (a, b, w[, attr])
    # and (b, a, w[, attr]).
    if attributes:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                # Equality -> loops
                if na <= nb:
                    yield (na, nb, ei.weight, ei.attributes)
    else:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                # Equality -> loops
                if na <= nb:
                    yield (na, nb, ei.weight)

def enumerate_edges_directed(adj_list, attributes=False):
    """
    Creates a generator that iterative through all the edges based on the give
    adjacency list, assuming a directed graph.  Each item will be a tuple
    of the format: (na, nb, w[, attr]).

    :param adj_list: Adjacency list.
    
    :param attributes: If set to true, edge attributes will be included.
    """
    if attributes:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                yield (na, nb, ei.weight, ei.attributes)
    else:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                yield (na, nb, ei.weight)

class GraphDataset:
    """
    Represents a graph dataset.
    Meant to be immutable so do not modify any of its data even though you can
    do that.

    Graph data are stored using the adjacency list. Therefore it is not memory
    efficient for dense graphs. Both node attributes and edge attributes are
    supported.

    Note: not suited for dense graphs.
    """

    def __init__(self, adj_list=[], weighted=False, directed=False,
                 name=None, node_labels=None, node_attributes=None, notes=''):
        """
        Creates a graph dataset.    
        """
        if name is None:
            name = 'Graph42'
        self._name = name
        self._notes = notes
        self._directed = directed
        self._weighted = weighted
        self._adj_list = adj_list
        # Node attributes
        if node_attributes is not None:
            if len(node_attributes) != len(adj_list):
                raise ValueError('The length of the node attribute list must be equal to the number of nodes.')
        self._node_attributes = node_attributes
        # Node labels
        if node_labels is not None:
            if len(node_labels) != len(adj_list):
                raise ValueError('The length of the node label list must be equal to the number of nodes.')
        self._node_labels = node_labels
        self._init_cache_members()
    
    def _init_cache_members(self):
        # Degree list
        self._deg_list = None
        self._in_deg_list = None
        # Loop detection
        self._has_self_loops = None

    @classmethod
    def from_edges(cls, n_nodes, edges, weighted=False, directed=False,
                   has_edge_attrs=False, **kwargs):
        """
        Creates a simple graph dataset from the given edge list.

        The edge list can be one of the following:
        1. A list, where each element is either a list or a tuple represeting an
           edge pair.
        2. A numpy array, where each row represents an edge pair.
        3. An iterator, where each item is either a list or a tuple representing
           an edge pair.
        The supplied edge attributes are assumed to be immutable.
        Upon construction the adjacency list will be built.
        The default parameters will create an empty graph.
        """
        # Adjacency list
        # node_id -> {neighbor_id : (weight, attributes)}
        # For undirected graphs, weights are always ones.
        # Each element in the adjacency list is a dictionary.
        if directed:
            adj_list = build_adj_list_directed(n_nodes, edges, weighted, has_edge_attrs)
        else:
            adj_list = build_adj_list_undirected(n_nodes, edges, weighted, has_edge_attrs)
        return cls(adj_list=adj_list, weighted=weighted, directed=directed, **kwargs)

    @property
    def name(self):
        """
        Retrieves the name of this graph dataset.
        """
        return self._name

    @property
    def notes(self):
        """
        Retrieves notes of this graph dataset.
        """
        return self._notes
            
    @property
    def adj_list(self):
        """
        Retrieves the adjacency list view.
        """
        return AdjacencyListView(self._adj_list)

    @property
    def deg_list(self):
        """
        Retrieves the out degree list.
        Do not modify.
        """
        if self._deg_list is None:
            # Compute on demand.
            self._deg_list = [len(l) for l in self._adj_list]
        return ListView(self._deg_list)

    @property
    def node_labels(self):
        return ListView(self._node_labels)

    @property
    def node_attributes(self):
        return ListView(self._node_attributes)

    @property
    def n_nodes(self):
        """
        Retrieves the total number of nodes.
        """
        return self.compute_n_nodes()

    @property
    def n_edges(self):
        """
        Retrieves the total number of edges.
        """
        return self.compute_n_edges()

    @property
    def n_max_edges(self, including_loops=False):
        """
        Returns the maximum number of edges.
        """
        n_nodes = self.n_nodes
        if self.directed:
            n = n_nodes * (n_nodes - 1)
        else:
            n = n_nodes * (n_nodes - 1) // 2
        if including_loops:
            n += n_nodes
        return n
    
    @property
    def weighted(self):
        """
        Returns if the graph is weighted.
        """
        return self._weighted

    @property
    def directed(self):
        """
        Returns if the graph is directed.
        """
        return self._directed

    @property
    def has_self_loops(self):
        """
        Returns if the graph has loop(s).
        """
        if self._has_self_loops is None:
            for i, l in enumerate(self.adj_list):
                if i in l:
                    self._has_self_loops = True
                    break
            self._has_self_loops = False
        return self._has_self_loops

    def __getstate__(self):
        return {
            'name': self._name,
            'weighted': self._weighted,
            'directed': self._directed,
            'adj_list': self._adj_list,
            'node_labels': self._node_labels,
            'node_attributes': self._node_attributes
        }

    def __setstate__(self, state):
        self._name = state['name']
        self._weighted = state['weighted']
        self._directed = state['directed']
        self._adj_list = state['adj_list']
        self._node_labels = state['node_labels']
        self._node_attributes = state['node_attributes']
        self._init_cache_members()

    def __getitem__(self, key):
        """
        Provides a convenient way of accessing nodes and edges.
        """
        if isinstance(key, tuple):
            na, nb = key
            check_node_index(na, self.n_nodes)
            check_node_index(nb, self.n_nodes)
            l = self._adj_list[na]
            if nb in l:
                return self._adj_list[na][nb]
            else:
                return None
        else:
            label = None if self._node_labels is None else self._node_labels[key]
            attr = None if self._node_attributes is None else self._node_attributes[key] 
            return NodeView(key, DictionaryView(self._adj_list[key]), label, attr)

    def get_edge_iter(self, attributes=False):
        """
        Returns a generator of edges (list of triplets).
        We return a generator here to avoid storing the whole edge list.
        """
        if self._directed:
            return enumerate_edges_directed(self._adj_list, attributes)
        else:
            return enumerate_edges_undirected(self._adj_list, attributes)

    def get_node_iter(self):
        """
        Returns a generator of nodes.
        """
        for i in range(self.n_nodes):
            yield self[i]

    def compute_n_nodes(self):
        # This is a general implementation.
        # For a specific data set, the number of nodes is usually known.
        # You can override this implementation in this case.
        return len(self._adj_list)

    def compute_n_edges(self):
        # This is a general implementation.
        # For a specific data set, the number of edges is usually known.
        # You can override this implementation in this case.
        n = sum(self.deg_list)
        if self.directed:
            return n
        else:
            # Need to consider loops.
            n_loops = 0
            for i, l in enumerate(self.adj_list):
                if i in l:
                    n_loops += 1
            return (n - n_loops) // 2 + n_loops

    def get_adj_matrix(self):
        """
        Generates the dense adjacency matrix.
        """
        A = np.zeros((self.n_nodes, self.n_nodes))
        if self.directed:
            for e in self.get_edge_iter():
                A[e[0], e[1]] = e[2]
        else:
            for e in self.get_edge_iter():
                A[e[0], e[1]] = e[2]
                A[e[1], e[0]] = e[2]
        return A

    def get_laplacian(self, normalize=False):
        """
        Computes the Laplacian matrix.
        """
        d = np.array(self.deg_list)
        L = np.diag(d) - self.get_adj_matrix()
        if normalize:
            if np.any(d == 0):
                raise Exception('Cannot have any isolated nodes.')
            d = np.reshape(1.0 / np.sqrt(d), (-1, 1))
            L = d * L * d.T
        return L

    def subgraph(self, nodes_to_remove=None, edges_to_remove=None, validate=False, name=None):
        """
        Returns a new dataset constructed from the subgraph after removing
        specified nodes and edges.

        This method will first remove edges and them remove nodes.

        Node attributes and labels will copied shallowly.
        
        :param nodes_to_remove: Nodes to be removed.
        :param edges_to_remove: Edges to be removed.
        :param validate: When set to True, will raise an error if any of the
        node/edge to be removed does not exist at any removal step (this implies
        that duplicates will result in an error). 
        """
        # Clones the adjacency list. Tuples storing the edge information are
        # shallowly copied because they are assumed immutable.
        adj_list = [x.copy() for x in self._adj_list]
        n_nodes = self.n_nodes
        if edges_to_remove is not None:
            for e in edges_to_remove:
                # Validate
                if e[0] < 0 or e[0] >= n_nodes or e[1] not in adj_list[e[0]]:
                    if validate:
                        raise ValueError('Edge {}-{} does not exist.'.format(e[0], e[1]))
                    else:
                        continue
                # Remove this edge
                del adj_list[e[0]][e[1]]
                # Do not forget self-loops
                if not self.directed and e[0] != e[1]:
                    del adj_list[e[1]][e[0]]
        if nodes_to_remove is not None:
            # Validate first
            if validate:
                for n in nodes_to_remove:
                    if n < 0 or n >= n_nodes:
                        raise ValueError('Node id {} is out of range.'.format(n))
            orig_len = len(nodes_to_remove)
            nodes_to_remove = set(nodes_to_remove)
            if validate and len(nodes_to_remove) != orig_len:
                raise ValueError('Attempting to remove a node more than once.')
            # Maps old nodes to new nodes
            new_id = 0
            node_map = [None] * n_nodes
            for i in range(n_nodes):
                if i in nodes_to_remove:
                    node_map[i] = -1
                    continue
                node_map[i] = new_id
                new_id += 1
            # Update the adjacency list
            new_n_nodes = n_nodes - len(nodes_to_remove)
            for na in range(n_nodes):
                if node_map[na] != -1:
                    # Remove nodes if necessary and update the keys according to
                    # the node id map
                    adj_list[node_map[na]] = {node_map[nb]: ei for nb, ei in adj_list[na].items() if nb not in nodes_to_remove}
            # Resize the list
            adj_list = adj_list[:new_n_nodes]
        # Create the new graph dataset
        if name is None:
            name = "Subgraph of " + self.name
        # Handle node attributes and labels
        # We will just do a shallow copy here
        if nodes_to_remove is not None:
            if self.node_attributes is not None:
                node_attributes = [self.node_attributes[i] for i in range(n_nodes) if node_map[i] != -1]
            else:
                node_attributes = None
            if self.node_labels is not None:
                node_labels = [self.node_labels[i] for i in range(n_nodes) if node_map[i] != -1]
            else:
                node_attributes = None
        else:
            node_attributes = self._node_attributes
            node_labels = self._node_labels
        sg = GraphDataset(adj_list=adj_list, weighted=self.weighted,
                          directed=self.directed, name=name,
                          node_attributes=node_attributes,
                          node_labels=node_labels)
        return sg

    def to_unweighted(self, name=None):
        """
        Creates an unweighted version of this graph.
        """
        if name is None:
            name = 'Unweighted version of ' + self.name
        if self.weighted:
            # Convert to unweighted.
            adj_list = []
            for d in self._adj_list:
                adj_list.append({k: EdgeInfo(1, ei.attributes) for k, ei in d.items()})
        else:
            # Nothing changed. Just a simple copy.
            adj_list = [x.copy() for x in self._adj_list]
        g = GraphDataset(adj_list=adj_list, weighted=False,
                         directed=self._directed, name=name,
                         node_attributes=self._node_attributes,
                         node_labels=self._node_labels)
        return g

    def networkx(self):
        """
        Converts to networkx's graph.
        """
        if networkx_installed:
            if self.directed:
                G = networkx.DiGraph()
            else:
                G = networkx.Graph()
            # Add nodes
            for i in range(self.n_nodes):
                G.add_node(i)
            if self._node_attributes is not None:
                for i in range(self.n_nodes):
                    G.nodes[i]['attributes'] = self._node_attributes[i]
            if self._node_labels is not None:
                for i in range(self.n_nodes):
                    G.nodes[i]['label'] = self._node_labels[i]
            # Add edges
            for e in self.get_edge_iter(True):
                if e[3] is not None:
                    G.add_edge(e[0], e[1], weight=e[2], attributes=e[3])
                else:
                    G.add_edge(e[0], e[1], weight=e[2])
            G.name = self.name
            return G
        else:
            raise Exception('networkx is not installed.')

    def summary(self):
        """
        Gets a summary string.
        """
        return "{0}: {1}, {2}, nodes: {3}, edges: {4}, max_deg: {5}, min_deg: {6}, notes: {7}".format(
            self.name,
            "weighted" if self.weighted else "unweighted",
            "directed" if self.directed else "undirected",
            self.n_nodes, self.n_edges, max(self.deg_list), min(self.deg_list),
            self.notes
        )


