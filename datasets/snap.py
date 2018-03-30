from graphz.dataset import GraphDataset
from datetime import datetime

def load_temporal(data_src, start_time=None, end_time=None, name=None, notes=None, **kwargs):
    """
    Base loader for SNAP temporal graph datasets such as
    http://snap.stanford.edu/data/sx-mathoverflow.html.
    Each line in the data file should be triplet of (SRC, TGT, UNIXTS)
    representing an edge:
        SRC: id of the source node (a user)
        TGT: id of the target node (a user)
        UNIXTS: Unix timestamp (seconds since the epoch)

    :param data_src: Specifies the location of the edge list file.

    :param start_time: Optional datetime object. All edges that are formed
    before start_time will not be included.

    :param end_time: Optional datetime object. All edges that are formed at
    of after end_time will not be included.
    """
    if start_time is None:
        start_time = datetime.min
    if end_time is None:
        end_time = datetime.max
    actual_start_time = datetime.max
    actual_end_time = datetime.min
    edges = []
    cur_node_id = 0
    node_id_map = {}
    # Construct the edge list
    with open(data_src, 'r') as f:
        for l in f:
            splits = l.rstrip('\n').split()
            na = int(splits[0])
            nb = int(splits[1])
            ts = datetime.fromtimestamp(int(splits[2]))
            if na not in node_id_map:
                node_id_map[na] = cur_node_id
                cur_node_id += 1
            if nb not in node_id_map:
                node_id_map[nb] = cur_node_id
                cur_node_id += 1
            if ts >= start_time and ts < end_time:
                edges.append((node_id_map[na], node_id_map[nb]))
                if ts < actual_start_time:
                    actual_start_time = ts
                if ts > actual_end_time:
                    actual_end_time = ts
    actual_start_time = actual_start_time
    actual_end_time = actual_end_time
    # Name and notes
    if name is None:
        name = 'Unnamed'
    if notes is None:
        if actual_end_time < actual_start_time:
            s = 'empty'
        else:
            s = '{} - {}'.format(
                actual_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                actual_end_time.strftime('%Y-%m-%d %H:%M:%S'))
        notes = 'timespan is ' + s
    return GraphDataset.from_edges(n_nodes=len(node_id_map), edges=edges, name=name, notes=notes, **kwargs)