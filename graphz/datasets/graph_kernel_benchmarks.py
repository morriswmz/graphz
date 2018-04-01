from graphz.settings import get_setting
from graphz.dataset import GraphDataset, MultiGraphDataset
from graphz.utils import download_file
import os
import numpy as np
from collections import namedtuple

# Benchmark Data Sets for Graph Kernels
# https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# It is encourged to cite the above website if you use their datasets.

LabelAttrTuple = namedtuple('LabelAttrTuple', ['label', 'attributes'])

def _load(dataset_name, data_dir=None, download_url=None):
    """
    General data loader.
    
    Let

    n = total number of nodes
    m = total number of edges
    N = number of graphs

    DS_A.txt (m lines): sparse (block diagonal) adjacency matrix for all graphs,
    each line corresponds to (row, col) resp. (node_id, node_id). **All graphs
    are undirected. Hence, DS_A.txt contains two entries for each edge.**
    DS_graph_indicator.txt (n lines): column vector of graph identifiers for all
    nodes of all graphs, the value in the i-th line is the graph_id of the node
    with node_id i.
    DS_graph_labels.txt (N lines): class labels for all graphs in the data set,
    the value in the i-th line is the class label of the graph with graph_id i.
    DS_node_labels.txt (n lines): column vector of node labels, the value in the
    i-th line corresponds to the node with node_id i.

    There are optional files if the respective information is available:

    DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt): labels for the
    edges in DS_A_sparse.txt.
    DS_edge_attributes.txt (m lines; same size as DS_A.txt): attributes for the
    edges in DS_A.txt.
    DS_node_attributes.txt (n lines): matrix of node attributes, the comma
    seperated values in the i-th line is the attribute vector of the node with
    node_id i.
    DS_graph_attributes.txt (N lines): regression values for all graphs in the
    data set, the value in the i-th line is the attribute of the graph
    with graph_id i.
    """

    if data_dir is None:
        data_dir = os.path.join(get_setting('data_dir'), dataset_name)

    # i-th line corresponds to the i-th edge: (na, nb). na and nb start from 1.
    edge_definition_filename = os.path.join(data_dir, dataset_name + '_A.txt')
    # i-th line corresponds to the i-th edge's label
    edge_label_filename = os.path.join(data_dir, dataset_name + '_edge_labels.txt')
    # i-th line corresponds to the i-th edge's attribute
    edge_attributes_filename = os.path.join(data_dir, dataset_name + '_edge_attributes.txt')
    # i-th line is the graph_id of the node with node_id i
    graph_indicator_filename = os.path.join(data_dir, dataset_name + '_graph_indicator.txt')
    # i-th line is the label of the i-th graph
    graph_label_filename = os.path.join(data_dir, dataset_name + '_graph_labels.txt')
    # i-th line is the label of the i-th node
    node_label_filename = os.path.join(data_dir, dataset_name + '_node_labels.txt')
    # i-th line is the attribute of the i-th node
    node_attributes_filename = os.path.join(data_dir, dataset_name + '_node_attributes.txt')

    # Download if possible
    if not os.path.exists(edge_definition_filename) and download_url is not None:
        print('Downloading data from ' + download_url)
        downloaded_file = download_file(download_url, get_setting('data_dir'), unpack=True)

    # Load graph labels
    with open(graph_label_filename, 'r') as f:
        graph_labels = [int(l) for l in f]
    n_graphs = len(graph_labels)

    # Load edges
    edges = []
    with open(edge_definition_filename, 'r') as f:
        for l in f:
            splits = l.split(',')
            # Convert to zero-based indexing.
            edges.append((int(splits[0]) - 1, int(splits[1]) - 1))
    
    # Load edge labels
    if os.path.exists(edge_label_filename):
        with open(edge_label_filename, 'r') as f:
            edge_labels = [int(l) for l in f]
        if len(edge_labels) != len(edges):
            raise Exception('The length of the edge label list does not match the number of edges.')
    else:
        edge_labels = None

    # Load edge attributes
    if os.path.exists(edge_attributes_filename):
        with open(edge_definition_filename, 'r') as f:
            edge_attributes = [tuple(map(float, l.split(','))) for l in f]
        if len(edge_attributes) != len(edges):
            raise Exception('The length of the edge attribute list does not match the number of edges.')
    else:
        edge_attributes = None

    # Combine edge labels and attributes into edge data tuples
    if edge_attributes is None:
        if edge_labels is not None:
            edge_data = [LabelAttrTuple(label, None) for label in edge_labels]
        else:
            edge_data = None
    else:
        if edge_labels is None:
            edge_data = [LabelAttrTuple(None, attr) for attr in edge_attributes]
        else:
            edge_data = [LabelAttrTuple(edge_labels[i], edge_attributes[i]) for i in range(len(edges))]

    # Load graph indicators
    with open(graph_indicator_filename, 'r') as f:
        # Convert to zero-based indexing.
        graph_indicators = np.array([int(l) - 1 for l in f])

    # Load node labels
    if os.path.exists(node_label_filename):
        with open(node_label_filename, 'r') as f:
            node_labels = [int(l) for l in f]
        if len(node_labels) != len(graph_indicators):
            raise Exception('The length of the node label list does not match the number of nodes.')
    else:
        node_labels = None

    # Load node attributes
    if os.path.exists(node_attributes_filename):
        with open(node_attributes_filename, 'r') as f:
            node_attributes = [tuple(map(float, l.split(','))) for l in f]
        if len(node_attributes) != len(graph_indicators):
            raise Exception('The length of the node attribute list does not match the number of nodes.')
    else:
        node_attributes = None

    # Construct the full graph
    if edge_data is None:
        g = GraphDataset.from_edges(n_nodes=len(graph_indicators),
            edges=edges, weighted=False, directed=False,
            node_labels=node_labels, node_attributes=node_attributes)
    else:
        # All graphs are undirected
        zipped = filter(lambda t : t[0][0] <= t[0][1], zip(edges, edge_data))
        edge_iter = map(lambda t: (t[0][0], t[0][1], 1, t[1]), zipped)
        g = GraphDataset.from_edges(n_nodes=len(graph_indicators),
            edges=edge_iter, weighted=False, directed=False, has_edge_data=True,
            node_labels=node_labels, node_attributes=node_attributes)
    
    # Extract subgraphs
    graphs = [None] * n_graphs
    for i in range(n_graphs):
        graphs[i] = g.subgraph(nodes_to_keep=np.nonzero(graph_indicators == i)[0], name=dataset_name + '-' + str(i))
    return MultiGraphDataset(graphs, graph_labels, dataset_name)

def load_mutag(data_dir=None):
    """
    Loads the MUTAG dataset from:
    https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
    Citation to the website is encouraged!

    The MUTAG dataset consists of 188 chemical compounds divided into two 
    classes according to their mutagenic effect on a bacterium. 

    The chemical data was obtained form http://cdb.ics.uci.edu and converted 
    to graphs, where vertices represent atoms and edges represent chemical 
    bonds. Explicit hydrogen atoms have been removed and vertices are labeled
    by atom type and edges by bond type (single, double, triple or aromatic).
    Chemical data was processed using the Chemistry Development Kit (v1.4).

    Node labels:

    0  C
    1  N
    2  O
    3  F
    4  I
    5  Cl
    6  Br

    Edge labels:

    0  aromatic
    1  single
    2  double
    3  triple

    Previous Use of the Dataset:
    * Kriege, N., Mutzel, P.: Subgraph matching kernels for attributed graphs.
      In: Proceedings of the 29th International Conference on Machine Learning
      (ICML-2012) (2012).

    References:
    * Debnath, A.K., Lopez de Compadre, R.L., Debnath, G., Shusterman, A.J., and
      Hansch, C. Structure-activity relationship of mutagenic aromatic and
      heteroaromatic nitro compounds. Correlation with molecular orbital
      energies and hydrophobicity. J. Med. Chem. 34(2):786-797 (1991).
    """
    return _load('MUTAG', data_dir=data_dir, download_url=r'https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip')
    
