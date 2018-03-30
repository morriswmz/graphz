from graphz.dataset import GraphDataset
from graphz.settings import get_setting
from graphz.utils import download_file

def load_data(data_src=None):
    """
    Loads the US power grid network (https://toreopsahl.com/datasets/#uspowergrid).

    :param data_src: Specifies the location of the edge list file you
    downloaded from http://opsahl.co.uk/tnet/datasets/USpowergrid_n4941.txt.
    """
    if data_src is None:
        data_dir = get_setting('data_dir')
        data_src = download_file('http://opsahl.co.uk/tnet/datasets/USpowergrid_n4941.txt', data_dir)

    def process_line(l):
        splits = l.rstrip('\n').split()
        return int(splits[0]) - 1, int(splits[1]) - 1

    with open(data_src, 'r') as f:
        edges = map(process_line, f)
        return GraphDataset.from_edges(n_nodes=4941, edges=edges, weighted=False,
                                       directed=False, name='US Power Grid')
