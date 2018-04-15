"""
Helper functions for reading .mtx files from 
http://networkrepository.com/format-info.php
"""

from scipy.io import mmread
from graphz.dataset import GraphDataset

def from_mtx_file(filename, weighted=True, directed=True, name=None):
    a = mmread(filename)
    return GraphDataset.from_adj_mat(a, weighted=weighted, directed=directed, name=name)