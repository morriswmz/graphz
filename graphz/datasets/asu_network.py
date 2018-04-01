from graphz.dataset import GraphDataset
from graphz.settings import get_setting
from graphz.utils import download_file
import os.path

def _load_asunetwork_dataset(data_dir, expected_n_nodes, expected_n_edges, **kwargs):
    """
    Processes the ASU network datasets:
    http://socialcomputing.asu.edu/pages/datasets

    :param data_dir: Directory containing nodes.csv, edges.csv, and group-edges.csv 
    """
    node_file = os.path.join(data_dir, 'nodes.csv')
    edge_file = os.path.join(data_dir, 'edges.csv')
    # Load nodes
    with open(node_file, 'r') as f:
        node_names = [l.rstrip('\n') for l in f]
        node_name_map = {x: i for i, x in enumerate(node_names)}
    assert(len(node_name_map) == expected_n_nodes)
    # Load edges
    edges = []
    with open(edge_file, 'r') as f:
        for l in f:
            splits = l.rstrip('\n').split(',')
            edges.append((node_name_map[splits[0]], node_name_map[splits[1]]))
    assert(len(edges) == expected_n_edges)
    
    # Process labels
    # Note: each node can have more than one label or zero label.
    label_file = os.path.join(data_dir, 'group-edges.csv')
    if os.path.exists(label_file):
        node_labels = [[] for i in range(expected_n_nodes)]
        with open(label_file, 'r') as f:
            for l in f:
                splits = l.rstrip('\n').split(',')
                nid = node_name_map[splits[0]]
                gid = int(splits[1])
                node_labels[nid].append(gid)
    else:
        node_labels = None
    return GraphDataset.from_edges(n_nodes=expected_n_nodes, edges=edges, **kwargs)

def load_blogcatalog3(data_dir=None):
    """
    Loads the BlogCatalog dataset (http://socialcomputing.asu.edu/datasets/BlogCatalog3).

    :param data_dir: Directory containing nodes.csv, edges.csv, and group-edges.csv 
    """
    return _load_asunetwork_dataset(data_dir, 10312, 333983, weighted=False, directed=False, name='BlogCatalog3')

def load_flickr(data_dir=None):
    """
    Loads the Flickr dataset (http://socialcomputing.asu.edu/datasets/Flickr).

    :param data_dir: Directory containing nodes.csv, edges.csv, and group-edges.csv 
    """
    return _load_asunetwork_dataset(data_dir, 80513, 5899882, weighted=False, directed=False, name='Flickr')

def load_youtube2(data_dir=None):
    """
    Loads the YouTube2 dataset (http://socialcomputing.asu.edu/datasets/YouTube2).

    :param data_dir: Directory containing nodes.csv, edges.csv, and group-edges.csv 
    """
    return _load_asunetwork_dataset(data_dir, 1138499, 2990443, weighted=False, directed=False, name='YouTube2')
