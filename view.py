from collections.abc import Sequence, Mapping

class ListView(Sequence):
    __slots__ = '_list'
    
    def __init__(self, l):
        """
        Wrappers a simple list and creates a view of this list.
        """
        self._list = l
    
    def __getitem__(self, key):
        return self._list[key]

    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        return iter(self._list)
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._list == other._list
        return False

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._list.__repr__())

class DictionaryView(Mapping):
    __slots__ = ['_dict']
    
    def __init__(self, d):
        """
        Wrappers a simple dictionary and creates a view of this dictionary.
        Note: this does not guarantee full read-only behavior. If any of the items
        in the dictionary mutable, it can still be changed.
        """
        self._dict = d
    
    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)
    
    def __iter__(self):
        return iter(self._dict)
    
    def __contains__(self, el):
        return el in self._dict
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._dict == other._dict
        return False
    
    def __hash__(self):
        # We make it unhashable here.
        return None

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._dict.__repr__())

class AdjacencyListView(ListView):
    
    def __getitem__(self, key):
        # Disable slicing.
        if isinstance(key, slice):
            raise KeyError('{} does not support slicing.'.format(self.__class__.__name__))
        # Remember to return a view.
        return DictionaryView(self._list[key])
    
    def __len__(self):
        return len(self._list)

    def __iter__(self):
        # Remember to return a view.
        for l in self._list:
            yield DictionaryView(l)
    
    def __hash__(self):
        # We make it unhashable here.
        return None




    