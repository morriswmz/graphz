from graphz.dataset import GraphDataset
from graphz.utils import reservoir_sampling
import numpy as np
import random
import warnings

class EdgeSampler:
    
    def __init__(self, graph):
        self.graph = graph

    def sample(self, n_samples):
        raise NotImplementedError()

class PositiveEdgeSampler(EdgeSampler):

    def sample(self, n_samples):
        """
        Samples from existing edges.
        """
        # Precheck
        if n_samples > self.graph.n_edges:
            raise ValueError('Number of positive samples cannot be greater than the total number of edges.')
        # Sample positives using reservoir sampling
        x = reservoir_sampling(self.graph.get_edge_iter(), n_samples)
        if not self.graph.directed:
            # For undirected graphs, we randoms exchanges the order of the two
            # nodes in the tuple to prevent learners from exploiting the
            # ordering infomation. We want our learners to be able to predict
            # links for both (na, nb) and (nb, na) as they represent the same
            # edge in an undirected graph
            x = [(e[1], e[0], 1) if random.random() >= 0.5 else (e[0], e[1], 1) for e in x]
        else:
            x = [(e[0], e[1], 1) for e in x]
        return x

class NegativeEdgeSampler(EdgeSampler):

    def sample(self, n_samples, exclusions=None):
        """
        Samples disconnected pairs (no self loops will be included).
        """
        # Precheck
        max_n_neg = self.graph.get_max_n_edges()
        if n_samples > max_n_neg - self.graph.n_edges:
            raise ValueError('Too many negative samples requested.')
        # Check the network sparsity level
        sparsity_level = (self.graph.n_edges + n_samples) / max_n_neg
        if sparsity_level > 0.05:
            warnings.warn('Graph is not sparse enough. Random sampling may be slow.')
        x = []
        # Sample negatives randomly
        if exclusions is not None and len(exclusions) > 0:
            if self.graph.directed:
                sampled_pairs = set(map(lambda e: (e[0], e[1]), exclusions))
            else:
                # For undirected graphs, (na, nb) and (nb, na) are equivalent.
                sampled_pairs = set()
                for e in exclusions:
                    if e[0] < e[1]:
                        sampled_pairs.add((e[0], e[1]))
                    else:
                        sampled_pairs.add((e[1], e[0]))
        else:   
            sampled_pairs = set()
        n_nodes = self.graph.n_nodes
        if self.graph.directed:
            for i in range(n_samples):
                while True:
                    na = random.randint(0, n_nodes - 1)
                    nb = random.randint(0, n_nodes - 1)
                    if na == nb or (nb in self.graph.adj_list[na]) or ((na, nb) in sampled_pairs):
                        continue
                    x.append((na, nb, 0))
                    sampled_pairs.add((na, nb))
                    break
        else:
            for i in range(n_samples):
                while True:
                    na = random.randint(0, n_nodes - 1)
                    nb = random.randint(0, n_nodes - 1)
                    # For undirected graphs, (na, nb) and (nb, na) correspond
                    # to the same edge when na != nb.
                    if na == nb:
                        # Ensure that na < nb when recording (na, nb) in sampled
                        # pairs so we won't sample an edge twice.
                        continue
                    if na > nb:
                        na, nb = nb, na
                    if (nb in self.graph.adj_list[na]) or ((na, nb) in sampled_pairs):
                        continue
                    # We randomly exchange na and nb here to prevent learners to
                    # exploit the fact that na < nb.
                    if random.random() >= 0.5:
                        x.append((na, nb, 0))
                    else:
                        x.append((nb, na, 0))
                    # When recording sampled pairs, always ensure that na < nb.
                    sampled_pairs.add((na, nb))
                    break
        return x

class BalancedEdgeSampler(EdgeSampler):

    def __init__(self, graph):
        """
        An edge sampler that samples equal number of positive and negative edges.
        """
        super().__init__(graph)
        self._pos_sampler = PositiveEdgeSampler(self.graph)
        self._neg_sampler = NegativeEdgeSampler(self.graph)

    def sample(self, n_samples):
        """
        Samples edges.

        :returns: List of triplets (na, nb, is_connected)
        """
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos
        # Sample
        x_pos = self._pos_sampler.sample(n_pos)
        x_neg = self._neg_sampler.sample(n_neg)
        return x_pos + x_neg
