from graphz.dataset import GraphDataset
from graphz.utils import download_file
from graphz.reader import from_pajek
from graphz.settings import get_setting

def load_usair97(data_src=None, ignore_weights=False):
    """
    US Air 97 dataset: http://vlado.fmf.uni-lj.si/pub/networks/data/map/USAir97.net
    from the pajek datasets: http://vlado.fmf.uni-lj.si/pub/networks/data/default.htm
    """
    if data_src is None:
        data_dir = get_setting('data_dir')
        data_src = download_file('http://vlado.fmf.uni-lj.si/pub/networks/data/map/USAir97.net', data_dir)
    return from_pajek(data_src, ignore_weights=ignore_weights)