import random
import os
from urllib.request import urlopen
from urllib.parse import urlparse
import shutil

def reservoir_sampling(iter, k):
    """
    Samples k elements from a stream of data using the reservoir sampling
    algorithm.
    """
    res = []
    n = 0
    for item in iter:
        n += 1
        if len(res) < k:
            res.append(item)
        else:
            idx = random.randint(0, n - 1)
            if idx < k:
                res[idx] = item
    return res

def download_file(url, save_dir, save_filename=None, unpack=False, verbose=True):
    """
    Downloads a file from the web.

    :return: Full file name of the downloaded file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_filename is None:
        parts = urlparse(url)
        save_filename = os.path.basename(parts.path)
    full_filename = os.path.join(save_dir, save_filename)
    if not os.path.exists(full_filename):
        # Download
        with urlopen(url) as res:
            with open(full_filename, 'wb') as f:
                shutil.copyfileobj(res, f, 8192)
        if verbose:
            print('Download file to {}'.format(full_filename))
        if unpack:
            unpack_file(full_filename, save_dir)
    return full_filename

def unpack_file(file_path, out_dir):
    """
    Unpacks a file.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = os.path.basename(file_path)
    fn, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == '.zip':
        import zipfile
        with zipfile.ZipFile(file_path) as f:
            f.extractall(out_dir)
    elif ext == '.gz':
        import gzip
        out_filename = os.path.join(out_dir, fn)
        with gzip.open(file_path, 'rb') as f_in:
            with open(out_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise Exception('Cannot unpack: {}'.format(file_path))
