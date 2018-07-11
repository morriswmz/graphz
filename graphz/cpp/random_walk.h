#pragma once
#include <pybind11/numpy.h>

namespace graphz
{
    /**
     * Generates randoms from a weighted graph.
     * @param neighbors Compressed neighbor list.
     * @param weights Compressed weight list.
     * @param offsets List of offsets for each neighbor list entry in neighbors.
     *      offsets[i + 1] - offsets[i] gives the degree of node i.
     *      neighbors[offsets[i]], neighbors[offsets[i] + 1], ...
     *      neighbors[offsets[i + 1] - offsets[i] - 1] are the neighbors of
     *      node i.
     * @param n_sims_per_node Specifies the number of random walks generated
     *      for each node.
     * @param n_steps_per_node Specifies the number of steps for each random
     *      walk.
     */
    pybind11::array_t<int> GenerateRandomWalksWeighted(
        pybind11::array_t<int> neighbors,
        pybind11::array_t<double> weights,
        pybind11::array_t<int> offsets,
        int n_sims_per_node,
        int n_steps_per_node);

    /**
     * Generates randoms from an unweighted graph.
     * @param neighbors Compressed neighbor list.
     * @param offsets List of offsets for each neighbor list entry in neighbors.
     *      offsets[i + 1] - offsets[i] gives the degree of node i.
     *      neighbors[offsets[i]], neighbors[offsets[i] + 1], ...
     *      neighbors[offsets[i + 1] - offsets[i] - 1] are the neighbors of
     *      node i.
     * @param n_sims_per_node Specifies the number of random walks generated
     *      for each node.
     * @param n_steps_per_node Specifies the number of steps for each random
     *      walk.
     */
    pybind11::array_t<int> GenerateRandomWalks(
        pybind11::array_t<int> neighbors,
        pybind11::array_t<int> offsets,
        int n_sims_per_node,
        int n_steps_per_node);

}






