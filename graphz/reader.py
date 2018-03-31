import re
import os
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
        return len(self._node_id_map)

def from_edge_list(filename, weighted=True, directed=True, name=None):
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
        n_nodes, edges = parse_edge_list(f)
        return GraphDataset.from_edges(n_nodes=n_nodes, edges=edges, weighted=weighted,
                                       directed=directed, name=name)

def parse_edge_list(lines, node_encoder=DefaultNodeEncoder()):
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
    edges = []
    for l in lines:
        if l.isspace():
            continue
        splits = l.split()
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
            w = 1.
        edges.append((na, nb, w))
    return node_encoder.count, edges
    

def from_adj_mat(filename, ignore_weights=False, as_undirected=False):
    pass

def parse_adj_mat(lines, ignore_weights=False, as_undirected=False):
    pass



def from_pajek(filename, ignore_weights=False):
    """
    A simple pajek file reader. Only support a single network.
    """
    with open(filename, 'r') as f:
        name, n_nodes, node_labels, edges, arcs = parse_pajek(f)
        if name is None:
            name = 'Unnamed'
        if len(arcs) == 0:
            # Simple undirected graph
            return GraphDataset.from_edges(n_nodes=n_nodes, edges=edges,
                weighted=not ignore_weights, directed=False, node_labels=node_labels)
        else:
            # Has directed edges
            if len(edges) > 0:
                # Merge
                for e in edges:
                    if e[0] != e[1]:
                        arcs.append((e[0], e[1], e[2]))
                        arcs.append((e[1], e[2], e[2]))
            else:
                edges = arcs
            return GraphDataset.from_edges(n_nodes=n_nodes, edges=edges,
                weighted=not ignore_weights, directed=True, node_labels=node_labels)
        
def parse_pajek(lines, ignore_weights=False):
    name = None
    line_iter = iter(lines)
    node_labels = None
    edges = None
    arcs = None
    skip = False
    cur_node_id = 0
    node_id_map = None
    while True:
        if not skip:
            try:
                l = next(line_iter).strip()
            except StopIteration:
                break
        else:
            skip = False
        if l.startswith('*Network'):
            if name is not None:
                raise Exception('Loading multiple networks is not supported yet.')
            name = l[8:].strip()
        elif l.startswith('*Vertices'):
            # Read all vertices
            n_nodes = int(l[9:].strip())
            node_id_map = {}
            node_labels = []
            for i in range(n_nodes):
                l = next(line_iter).strip()
                nid, nlabel = _parse_pajek_vertex_line(l)
                if nid not in node_id_map:
                    node_id_map[nid] = cur_node_id
                    cur_node_id += 1
                    node_labels.append(nlabel)
                else:
                    raise Exception('Duplicate vertex definition: ' + l)
        elif l.startswith('*Edges'):
            # Read all edges (undirected)
            # Format: *Edges :n "relation name"
            if node_id_map is None:
                raise Exception('Expecting vertex section first.')
            if edges is not None:
                raise Exception('No multiple edge section support yet.')
            edges = []
            while line_iter:
                try:
                    l = next(line_iter).strip()
                except StopIteration:
                    break
                if l.startswith('*'):
                    skip = True
                    break
                edges.append(_parse_pajek_edge_line(l, node_id_map, ignore_weights))
        elif l.startswith('*Arcs'):
            # Read all arcs (directed)
            # Format: *Arcs :n "relation name"
            if node_id_map is None:
                raise Exception('Expecting vertex section first.')
            if arcs is not None:
                raise Exception('No multiple arc section support yet.')
            arcs = []
            while line_iter:
                try:
                    l = next(line_iter).strip()
                except StopIteration:
                    break
                if l.startswith('*'):
                    skip = True
                    break
                arcs.append(_parse_pajek_edge_line(l, node_id_map, ignore_weights))
        else:
            if l.isspace() > 0:
                raise Exception('Unexpected syntax: ' + l)
    # Normalize return values. No Nones
    if edges is None:
        edges = []
    if arcs is None:
        arcs = []
    return name, n_nodes, node_labels, edges, arcs

pajek_vertex_re = re.compile(r'^\s*(\d+)\s+"(.+)"')
pajek_edge_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+([\d\.]+)')

def _parse_pajek_vertex_line(l):
    # Format: id "label" coordX coordY value shape factX factY color [activ_int]
    # We only retrieve id and label here
    m = pajek_vertex_re.match(l)
    return int(m.group(1)), m.group(2)

def _parse_pajek_edge_line(l, node_id_map, ignore_weights):
    # Format: init_vertex term_vertex value width color [activ_int]
    # We only retrieve init_vertex, term_vertex, and value here
    m = pajek_edge_re.match(l)
    na, nb, w = int(m.group(1)), int(m.group(2)), m.group(3)
    na = node_id_map[na]
    nb = node_id_map[nb]
    if ignore_weights:
        return na, nb, 1
    else:
        return na, nb, float(w)
