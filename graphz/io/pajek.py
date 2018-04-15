import re
from graphz.dataset import GraphDataset

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
