from collections import defaultdict

defaults = defaultdict(str)
defaults['data_dir'] = 'data'

def get_setting(key):
    return defaults[key]