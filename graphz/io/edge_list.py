import os
from collections import namedtuple
from graphz.dataset import GraphDataset

class DefaultNodeEncoder:
    
    def __init__(self):
        """
        Creates an encoder that encodes node names by their order of appearance.
        """
        self._node_id_map = {}
        self._id_node_map = []
        self._cur_node_id = 0

    def encode(self, nn):
        """
        Encodes a node name.
        """
        if nn not in self._node_id_map:
            self._node_id_map[nn] = self._cur_node_id
            self._id_node_map.append(nn)
            self._cur_node_id += 1
        return self._node_id_map[nn]

    def decode(self, id):
        """
        Decodes a node name.
        """
        return self._id_node_map[id]

    @property
    def count(self):
        """
        Returns the number of nodes.
        """
        return len(self._node_id_map)

def from_edge_list(filename, weighted=True, directed=True, name=None,
                   delimiter=None, comment_line_start=None):
    """
    Construct a undirected graph from a text file with simple edge lists:
    1 2 0.1
    2 3 0.1
    ...
    
    Note: the node name can be strings without whitespace in them. For instance,
    n1 n2 0.1
    n2 n3 0.2
    ...
    However, these names will be encoded into integers.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, 'r') as f:
        n_nodes, edges, node_labels = parse_edge_list(f, delimiter, comment_line_start)
        return GraphDataset.from_edges(n_nodes=n_nodes, edges=edges, weighted=weighted,
                                       directed=directed, name=name, node_labels=node_labels)

def parse_edge_list(lines, delimiter=None, comment_line_start=None, node_encoder=None):
    """
    Parses lines of edge definitions.

    By default, the parser will re-encode each node by their order of appearance
    when constructing the graph. For instance, the following edge list
    1 3 0.5
    1 2 0.1
    will turn into
    (0, 1, 0.5), (0, 2, 0.1)
    Here 1 appears first and is encoded as 0.
    3 appears second and is encoded as 1.
    2 appears third and is encoded as 2.

    :returns: A list of edge tuples: (na, nb, w).
    """
    if node_encoder is None:
        node_encoder = DefaultNodeEncoder()
    edges = []
    for l in lines:
        if l.isspace() or (comment_line_start is not None and l.startswith(comment_line_start)):
            continue
        splits = l.split(delimiter)
        if len(splits) < 2 or len(splits) > 3:
            raise Exception('Invalid edge definition at: ' + l)
        # Encode node ids.
        na_str, nb_str = splits[0], splits[1]
        na = node_encoder.encode(na_str)
        nb = node_encoder.encode(nb_str)
        # default weight is 1
        if len(splits) == 3:
            w = float(splits[2])
        else:
            w = 1
        edges.append((na, nb, w))
    node_labels = [node_encoder.decode(i) for i in range(node_encoder.count)]
    return node_encoder.count, edges, node_labels
    