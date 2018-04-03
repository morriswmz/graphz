import numpy as np
from scipy import sparse as sp
import copy
from collections import namedtuple, defaultdict
from graphz.view import DictionaryView, ListView, AdjacencyListView
try:
    import networkx
    networkx_installed = True
except ImportError:
    networkx_installed = False

"""
Stores information of an edge in adjacency lists.
"""
EdgeInfo = namedtuple('EdgeInfo', ['weight', 'data'])

"""
Provides full information of a node.
"""
NodeView = namedtuple('NodeView', ['id', 'out_edges', 'label', 'attributes'])

# Represents an edge with unit weight and no data.
UW_NA_EDGE_INFO = EdgeInfo(1, None) 

def check_node_index(nid, n_nodes):
    if nid < 0 or nid >= n_nodes:
        raise ValueError('Node id {} is out of bounds.'.format(nid))

def merge_edge(na, nb, eia, eib):
    pass

def build_adj_list_undirected(n_nodes, edges, weighted=False, has_data=False):
    """
    Builds a adjacency list from an edge list for an undirected graph.

    :param n_nodes: Number of nodes.
    
    :param edges: Collection of edges. Each item must have the following
    structure: (na, nb[, w, attr]). When weighted is True, w must be present.
    When has_data is True, both w and attr must be present (even if w will
    be ignored if weighted is False). Examples:
    1. weighted=False, has_data=False
        edges = [(0, 1), (0, 2), (1, 2)]
    2. weighted=True, has_data=False
        edges = [(0, 1, 0.2), (0, 2, 0.4), (1, 2, 0.3)]
    3. weighted=False, has_data=True
        edges = [(0, 1, 1, 'e1'), (0, 2, 1, 'e2'), (1, 2, 1, 'e3')]
    4. weighted=True, has_data=True
        edges = [(0, 1, 0.2, 'e1'), (0, 2, 0.4, 'e2'), (1, 2, 0.3, 'e3')]
    Note: when has_data is True, edges cannot contain duplicate definitions.
    
    :param weighted: Specifies whether the edges have weights.

    :param has_data: Specified whether the edges have data.

    :return: The constructed adjacency list.
    """
    adj_list = [{} for i in range(n_nodes)]
    if weighted:
        if has_data:
            # Data provided.
            # If an edge pair appear multiple times, an error will be raised
            # because we do not know how to merge the data.
            for e in edges:
                na, nb, w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                # For undirected graphs, adj_list[na][nb] and adj_list[nb][na]
                # are updated simultaneously. We only need to check one of them.
                if nb in adj_list[na]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when data are provided.'.format(na, nb))
                ei = EdgeInfo(w, attr)
                adj_list[na][nb] = ei
                adj_list[nb][na] = ei
        else:
            # No data provided.
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
        if has_data:
            for e in edges:
                na, nb, _w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if nb in adj_list[na]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when data are provided.'.format(na, nb))
                ei = EdgeInfo(1, attr)
                adj_list[na][nb] = ei
                adj_list[nb][na] = ei
        else:
            # When no edge data are present, duplications are ignored for
            # undirected graphs.
            for e in edges:
                na, nb = e[0], e[1]
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                adj_list[na][nb] = UW_NA_EDGE_INFO
                adj_list[nb][na] = UW_NA_EDGE_INFO
    return adj_list

def build_adj_list_directed(n_nodes, edges, weighted=False, has_data=False):
    """
    Builds a adjacency list from an edge list for a directed graph.

    :param n_nodes: Number of nodes.
    
    :param edges: Collection of edges. Each item must have the following
    structure: (na, nb[, w, attr]). When weighted is True, w must be present.
    When has_data is True, both w and attr must be present (even if w will
    be ignored if weighted is False). Examples:
    1. weighted=False, has_data=False
        edges = [(0, 1), (0, 2), (1, 2)]
    2. weighted=True, has_data=False
        edges = [(0, 1, 0.2), (0, 2, 0.4), (1, 2, 0.3)]
    3. weighted=False, has_data=True
        edges = [(0, 1, 1, 'e1'), (0, 2, 1, 'e2'), (1, 2, 1, 'e3')]
    4. weighted=True, has_data=True
        edges = [(0, 1, 0.2, 'e1'), (0, 2, 0.4, 'e2'), (1, 2, 0.3, 'e3')]
    Note: when has_data is True, edges cannot contain duplicate definitions.
    
    :param weighted: Specifies whether the edges have weights.

    :param has_data: Specified whether the edges have data.

    :return: The constructed adjacency list.
    """
    adj_list = [{} for i in range(n_nodes)]
    if weighted:
        if has_data:
            # Data provided.
            # If an edge pair appear multiple times, an error will be raised
            # because we do not know how to merge the data.
            for e in edges:
                na, nb, w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if nb in adj_list[na]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when data are provided.'.format(na, nb))
                adj_list[na][nb] = EdgeInfo(w, attr)
        else:
            # No data provided.
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
        if has_data:
            for e in edges:
                na, nb, _w, attr = e
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                if na in adj_list[nb]:
                    raise ValueError('Cannot merge duplicate edge definitions for the edge {}->{} when data are provided.'.format(na, nb))
                adj_list[na][nb] = EdgeInfo(1, attr)
        else:
            # When no edge data are present, duplications are ignored for
            # undirected graphs.
            for e in edges:
                na, nb = e[0], e[1]
                check_node_index(na, n_nodes)
                check_node_index(nb, n_nodes)
                adj_list[na][nb] = UW_NA_EDGE_INFO
    return adj_list

def enumerate_edges_undirected(adj_list, data=False):
    """
    Creates a generator that iterative through all the edges based on the give
    adjacency list, assuming an undirected graph. Each item will be a tuple
    of the format: (na, nb, w[, attr]).

    :param adj_list: Adjacency list.
    
    :param data: If set to true, edge data will be included.
    """
    # For undirected graphs, we do not want to record both (a, b, w[, attr])
    # and (b, a, w[, attr]).
    if data:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                # Equality -> loops
                if na <= nb:
                    yield (na, nb, ei.weight, ei.data)
    else:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                # Equality -> loops
                if na <= nb:
                    yield (na, nb, ei.weight)

def enumerate_edges_directed(adj_list, data=False):
    """
    Creates a generator that iterative through all the edges based on the give
    adjacency list, assuming a directed graph.  Each item will be a tuple
    of the format: (na, nb, w[, attr]).

    :param adj_list: Adjacency list.
    
    :param data: If set to true, edge data will be included.
    """
    if data:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                yield (na, nb, ei.weight, ei.data)
    else:
        for na, l in enumerate(adj_list):
            for nb, ei in l.items():
                yield (na, nb, ei.weight)

class GraphDataset:
    """
    Represents a graph dataset. Suitable for experiments using a single graph,
    such as node embedding and link prediction.
    Meant to be immutable.

    Graph data are stored using the adjacency list. Therefore it is not memory
    efficient for dense graphs. Both node attributes and edge data are
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
        n_nodes = len(adj_list)
        if node_attributes is not None:
            if len(node_attributes) != n_nodes:
                raise ValueError('The length of the node attribute list must be equal to the number of nodes.')
            self._node_attributes = node_attributes
        # Node labels
        if node_labels is not None:
            if len(node_labels) != n_nodes:
                raise ValueError('The length of the node label list must be equal to the number of nodes.')
            self._node_labels = node_labels

    @classmethod
    def from_edges(cls, n_nodes, edges, weighted=False, directed=False,
                   has_edge_data=False, **kwargs):
        """
        Creates a simple graph dataset from the given edge list.

        The edge list can be one of the following:
        1. A list, where each element is either a list or a tuple represeting an
           edge pair.
        2. A numpy array, where each row represents an edge pair.
        3. An iterator, where each item is either a list or a tuple representing
           an edge pair.
        The supplied edge data are assumed to be immutable.
        Upon construction the adjacency list will be built.

        Edge data and node attributes/labels should be immutable.
        """
        # Adjacency list
        # node_id -> {neighbor_id : (weight, data)}
        # For undirected graphs, weights are always ones.
        # Each element in the adjacency list is a dictionary.
        if directed:
            adj_list = build_adj_list_directed(n_nodes, edges, weighted, has_edge_data)
        else:
            adj_list = build_adj_list_undirected(n_nodes, edges, weighted, has_edge_data)
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
        try:
            deg_list = self._deg_list
        except AttributeError:
            # Compute on demand.
            deg_list = [len(l) for l in self._adj_list]
            self._deg_list = deg_list
        return ListView(deg_list)

    @property
    def node_labels(self):
        try:
            node_labels = self._node_labels
        except AttributeError:
            raise Exception('No node labels are available.')
        return ListView(node_labels)

    @property
    def node_attributes(self):
        try:
            node_attributes = self._node_attributes
        except AttributeError:
            raise Exception('No node attributes are available.')
        return ListView(node_attributes)

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
        try:
            has_self_loops = self._has_self_loops
        except AttributeError:
            for i, l in enumerate(self.adj_list):
                if i in l:
                    has_self_loops = True
                    break
            has_self_loops = False
            self._has_self_loops = has_self_loops
        return has_self_loops

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
            try:
                label = self._node_labels[key]
            except AttributeError:
                label = None
            try:
                attr = self._node_attributes[key]
            except AttributeError:
                attr = None
            return NodeView(key, DictionaryView(self._adj_list[key]), label, attr)

    def get_n_max_edges(self, including_loops=False):
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

    def get_edge_iter(self, data=False):
        """
        Returns a generator of edges (list of triplets).
        We return a generator here to avoid storing the whole edge list.
        """
        if self._directed:
            return enumerate_edges_directed(self._adj_list, data)
        else:
            return enumerate_edges_undirected(self._adj_list, data)

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

    def get_adj_matrix(self, sparse=False):
        """
        Generates the dense adjacency matrix.
        """
        if sparse:
            A = sp.dok_matrix((self.n_nodes, self.n_nodes))
        else:
            A = np.zeros((self.n_nodes, self.n_nodes))
        if self.directed:
            for e in self.get_edge_iter():
                A[e[0], e[1]] = e[2]
        else:
            for e in self.get_edge_iter():
                A[e[0], e[1]] = e[2]
                A[e[1], e[0]] = e[2]
        if sparse:
            return A.tocsr()
        else:
            return A

    def get_laplacian(self, sparse=False, normalize=False):
        """
        Computes the Laplacian matrix.
        """
        if sparse:
            A = self.get_adj_matrix(True)
            d = np.asarray(A.sum(1)).squeeze()
            L = sp.diags(d) - A
            if normalize:
                D = sp.diags(np.reciprocal(np.sqrt(d)))
                L = D @ L @ D
        else:
            A = self.get_adj_matrix(False)
            d = A.sum(1)
            L = np.diag(d) - A
            if normalize:
                d = np.reshape(1.0 / np.sqrt(d), (-1, 1))
                L = d * L * d.T
        return L

    def subgraph(self, nodes_to_remove=None, nodes_to_keep=None, edges_to_remove=None, validate=False, name=None):
        """
        Returns a new dataset constructed from the subgraph after removing
        specified nodes and edges.

        This method will first remove edges and them remove nodes.

        Node attributes and labels will copied shallowly.
        
        :param nodes_to_remove: Nodes to be removed. Mutually exclusive with
        nodes_to_keep.
        :param nodes_to_keep: Nodes to be kept. Mutually exclusive with
        nodes_to_remove.
        :param edges_to_remove: Edges to be removed.
        :param validate: When set to True, will raise an error if any of the
        node/edge to be removed does not exist at any removal step (this implies
        that duplicates will result in an error). 
        """
        if nodes_to_remove is not None and nodes_to_keep is not None:
            raise ValueError('nodes_to_remove and nodes_to_keep are mutually exclusive.')
        # Construct id map between old nodes and new nodes.
        n_nodes = self.n_nodes
        if nodes_to_remove is not None:
            node_map = self._sg_build_node_map_exclusive(nodes_to_remove, validate)
        elif nodes_to_keep is not None:
            node_map = self._sg_build_node_map_inclusive(nodes_to_keep, validate)
        else:
            node_map = None
        # Construct the new adjacency list according to the node id map.
        adj_list = self._sg_build_new_adj_list(node_map)
        # Remove edges
        self._sg_remove_edges(adj_list, node_map, edges_to_remove, validate)
        # Create the new graph dataset
        if name is None:
            name = "Subgraph of " + self.name
        # Handle node attributes and labels
        new_node_attributes, new_node_labels = self._sg_slice_node_data(node_map)
        sg = GraphDataset(adj_list=adj_list, weighted=self.weighted,
                          directed=self.directed, name=name,
                          node_attributes=new_node_attributes,
                          node_labels=new_node_labels)
        return sg

    def _sg_build_node_map_inclusive(self, nodes_to_keep, validate):
        """
        Returns a map between the set of original ids of the remaining nodes in
        the original graph and their new ids.
        """
        n_nodes = self.n_nodes
        new_id = 0
        node_map = {}
        for n in nodes_to_keep:
            if n in node_map:
                if validate:
                    raise ValueError('Attempting to keep a node twice.')
                continue
            if n < 0 or n >= n_nodes:
                if validate:
                    raise ValueError('Node id {} is out of bounds.'.format(n))
                continue
            node_map[n] = new_id
            new_id += 1
        return node_map

    def _sg_build_node_map_exclusive(self, nodes_to_remove, validate):
        """
        Returns a map between the set of original ids of the remaining nodes in
        the original graph and their new ids.
        """
        n_nodes = self.n_nodes        
        ex_set = set()
        for n in nodes_to_remove:
            if n in ex_set:
                if validate:
                    raise ValueError('Attempting to remove a node twice.')
                continue
            if n < 0 or n >= n_nodes:
                if validate:
                    raise ValueError('Node id {} is out of bounds.'.format(n))
                continue
            ex_set.add(n)
        new_id = 0
        node_map = {}
        for i in range(n_nodes):
            if i in ex_set:
                continue
            node_map[i] = new_id
            new_id += 1
        return node_map

    def _sg_build_new_adj_list(self, node_map):
        """
        Builds the new adjacency list for the subgraph according to the
        node id map.
        """
        if node_map is None or len(node_map) == self.n_nodes:
            # No node is removed. Simply return a copy.
            adj_list = [x.copy() for x in self._adj_list]
        else:
            # Need some adjustments
            adj_list = [None] * len(node_map)
            for old_id, new_id in node_map.items():
                new_dict = {}
                for nb, e in self._adj_list[old_id].items():
                    if nb in node_map:
                        new_dict[node_map[nb]] = e
                adj_list[new_id] = new_dict
        return adj_list

    def _sg_remove_edges(self, adj_list, node_map, edges_to_remove, validate):
        """
        Remove edges.
        """
        # We have to keep in mind that the dictionary keys in the new adj_list
        # have been adjusted, while edges_to_remove still uses node ids from the
        # original graph.
        if edges_to_remove is None:
            return
        n_nodes = self.n_nodes
        for e in edges_to_remove:
            # Validate
            if e[0] < 0 or e[0] >= n_nodes or e[1] not in self._adj_list[e[0]]:
                if validate:
                    raise ValueError('Edge {}-{} does not exist in the original graph.'.format(e[0], e[1]))
                continue
            # Remove this edge
            if node_map is None:
                new_id_a = e[0]
                new_id_b = e[1]
            else:
                # e[0] and e[1] correspond to old node ids
                if e[0] in node_map and e[1] in node_map:
                    new_id_a = node_map[e[0]]
                    new_id_b = node_map[e[1]]
                else:
                    # This edge no longer exists
                    continue
            if new_id_b not in adj_list[new_id_a]:
                if validate:
                    raise ValueError('Edge {}-{} is already removed.'.format(e[0], e[1]))
                continue
            del adj_list[new_id_a][new_id_b]
            # Do not forget self-loops
            if not self.directed and new_id_a != new_id_b:
                del adj_list[new_id_b][new_id_a]
    
    def _sg_slice_node_data(self, node_map):
        """
        Prepare the node data for the subgraph.
        """
        # We will just do a shallow copy here.
        if node_map is None:
            # Nodes remain unchanged.
            return self._node_attributes, self._node_labels
        
        try:
            node_attributes = self._node_attributes
        except AttributeError:
            node_attributes = None
        try:
            node_labels = self._node_labels
        except AttributeError:
            node_labels = None
        if node_attributes is not None:
            new_node_attributes = [None] * len(node_map)
            for old_i, new_i in node_map.items():
                new_node_attributes[new_i] = node_attributes[old_i]
        else:
            new_node_attributes = None
        
        if node_labels is not None:
            new_node_labels = [None] * len(node_map)
            for old_i, new_i in node_map.items():
                new_node_labels[new_i] = node_labels[old_i]
        else:
            new_node_labels = None
        return new_node_attributes, new_node_labels
        
    def _process_nodes_input_for_subgraph(self, nodes, validate):
        # Validate first
        if validate:
            for n in nodes:
                if n < 0 or n >= self.n_nodes:
                    raise ValueError('Node id {} is out of range.'.format(n))
        orig_len = len(nodes)
        nodes = set(nodes)
        if validate and len(nodes) != orig_len:
            raise ValueError('Duplicated node ids are not allowed.')
        return nodes

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
                adj_list.append({k: EdgeInfo(1, ei.data) for k, ei in d.items()})
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
            # Copy node data
            try:
                node_attributes = self._node_attributes
            except AttributeError:
                node_attributes = None
            if node_attributes is not None:
                for i in range(self.n_nodes):
                    G.nodes[i]['attributes'] = node_attributes[i]
            try:
                node_labels = self._node_labels
            except AttributeError:
                node_labels = None
            if node_labels is not None:
                for i in range(self.n_nodes):
                    G.nodes[i]['label'] = node_labels[i]
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

class MultiGraphDataset:
    """
    A dataset containing a collection of graphs. Suitable for experiments
    involving many graphs such as graph classification/matching.
    Meant to be immutable.
    """
    def __init__(self, graphs, labels=None, name='Graphs'):
        """
        Initializes a multi-graph dataset with a collection of graphs.

        :param graphs: A list/iterator of graphs.
        :param labels: A list of labels for the graphs.
        """
        self._graphs = list(graphs) # Ensure a list here.
        if labels is not None:
            if len(labels) != len(self._graphs):
                raise Exception('The length of label list must match that of the graph list.')
        else:
            labels = [None] * len(self._graphs)
        self._graph_labels = labels
        self._n_classes = None
        self._name = name
    
    @property
    def name(self):
        return self._name

    @property
    def n_graphs(self):
        """
        Retrieves the number of graphs.
        """
        return len(self._graphs)

    @property
    def n_classes(self):
        """
        Retrives the number of classes.
        """
        if self._graph_labels is None:
            return 0
        if self._n_classes is None:
            self._n_classes = len(set(self._graph_labels))
        return self._n_classes

    @property
    def graphs(self):
        return ListView(self._graphs)

    @property
    def labels(self):
        return ListView(self._graph_labels)

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, key):
        if self._graph_labels is not None:
            return self._graphs[key], self._graph_labels[key]
        else:
            return self._graphs[key], None

    def summary(self):
        return '{0}: n_graphs={1}, n_classes={2}, avg_n_nodes={3:.2f}, avg_n_edges={4:.2f}'.format(
            self.name, self.n_graphs, self.n_classes,
            np.mean([g.n_nodes for g in self._graphs]),
            np.mean([g.n_edges for g in self._graphs])
        )
