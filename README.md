# graphz

Temporary place for my helper module which I use in my graph related experiments.

Mainly used to load various graph datasets (do remember to cite the original authors if you use their datasets). A dataset with a single graph (for experiments related to node embedding, link prediction, etc.) is loaded as a `GraphDataset`, which is meant to be immutable. Both node attributes and edge attributes are supported. There is also support for datasets with multiple graphs (for graph classification experiments).

Graphs are stored using adjacency lists. Some basic analysis methods are implemented. For more serious analysis, it is recommended to more developed tools such as [networkx](https://networkx.github.io/). You can simply call `networkx()` on a `GraphDataset` object to convert it to a `networkx` graph. 
