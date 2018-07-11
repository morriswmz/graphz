#include <iostream>
#include <stdexcept>
#include <random>
#include <omp.h>

#include "array_slice.h"
#include "random_walk.h"
#include "sampling.h"

namespace graphz
{
    void GenerateRandomWalksImpl(
        int *neighbors,
        int *offsets,
        int *output,
        int n_nodes,
        int n_sims_per_node,
        int n_steps_per_node)
    {
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 generator(rd());
            #pragma omp master
            {
                std::cout << "Using " << omp_get_num_threads() << " thread(s)." << std::endl;
            }
            #pragma omp for
            for (int node_id = 0; node_id < n_nodes; node_id++)
            {
                for (int i = 0; i < n_sims_per_node; i++)
                {
                    // Performs a single random walk.
                    // Because each thread writes to non-overlapping locations,
                    // no lock is required.
                    auto cur_node_id = node_id;
                    auto row_offset = node_id * n_sims_per_node + i;
                    output[row_offset * n_steps_per_node] = cur_node_id;
                    for (int step = 1; step < n_steps_per_node; step++)
                    {
                        auto deg = offsets[cur_node_id + 1] - offsets[cur_node_id];
                        if (deg > 0)
                        {
                            // Only update for non-isolated node
                            auto next_node_offset = std::uniform_int_distribution<int>{ 0, deg - 1 }(generator);
                            cur_node_id = neighbors[offsets[cur_node_id] + next_node_offset];
                        }
                        output[row_offset * n_steps_per_node + step] = cur_node_id;
                    }
                }
            }
        }
    }

    void GenerateRandomWalksImpl(
        int *neighbors,
        double *weights,
        int *offsets,
        int *output,
        int n_nodes,
        int n_sims_per_node,
        int n_steps_per_node)
    {
        // Prepare the alias table.
        auto n_edges = offsets[n_nodes];
        auto ptr_p = new double[n_edges];
        auto ptr_q = new int[n_edges];
        for (int i = 0;i < n_nodes;i++)
        {
            CreateAliasTable(
                offsets[i + 1] - offsets[i],
                weights + offsets[i],
                ptr_p + offsets[i],
                ptr_q + offsets[i]);
        }
        std::cout << "edges: " << n_edges << std::endl;
        for (int i = 0;i < n_edges;i++)
        {
            std::cout << i << ':' << weights[i] << '-' << ptr_p[i] << '(' << ptr_q[i] << ')' << std::endl;
        }
        // Random Walk
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 generator(rd());
            #pragma omp for
            for (int node_id = 0; node_id < n_nodes; node_id++)
            {
                for (int i = 0; i < n_sims_per_node; i++)
                {
                    // Performs a single random walk.
                    // Because each thread writes to non-overlapping locations,
                    // no lock is required.
                    auto cur_node_id = node_id;
                    auto row_offset = node_id * n_sims_per_node + i;
                    output[row_offset * n_steps_per_node] = cur_node_id;
                    for (int step = 1; step < n_steps_per_node; step++)
                    {
                        auto deg = offsets[cur_node_id + 1] - offsets[cur_node_id];
                        if (deg > 0)
                        {
                            // Only update for non-isolated node
                            auto next_node_offset = AliasSample(deg, ptr_p + offsets[cur_node_id], ptr_q + offsets[cur_node_id], generator);
                            cur_node_id = neighbors[offsets[cur_node_id] + next_node_offset];
                        }
                        output[row_offset * n_steps_per_node + step] = cur_node_id;
                    }
                }
            }
        }
        // Clean up
        delete ptr_p;
        delete ptr_q;
    }

    pybind11::array_t<int> GenerateRandomWalksWeighted(
        pybind11::array_t<int> neighbors,
        pybind11::array_t<double> weights,
        pybind11::array_t<int> offsets,
        int n_sims_per_node,
        int n_steps_per_node)
    {
        // Input processing.
        auto bi_neighbors = neighbors.request();
        if (bi_neighbors.ndim != 1)
        {
            throw std::runtime_error("Neighbor list should be an 1D array.");
        }
        auto bi_weights = weights.request();
        if (bi_weights.size != 0 && bi_weights.shape != bi_neighbors.shape)
        {
            throw std::runtime_error("Weight list must have the same shape as the neighbor list.");
        }
        auto bi_offsets = offsets.request();
        if (bi_offsets.ndim != 1)
        {
            throw std::runtime_error("Offset list should be an 1D array.");
        }
        auto n_nodes = bi_offsets.size - 1;
        // Allocate the output array.
        std::vector<size_t> output_shape {
            static_cast<size_t>(n_sims_per_node * n_nodes),
            static_cast<size_t>(n_steps_per_node)
        };
        pybind11::array_t<int> output(output_shape);
        // Generate.
        auto ptr_neighbors = static_cast<int *>(bi_neighbors.ptr);
        auto ptr_weights = bi_weights.size == 0 ? nullptr : static_cast<double *>(bi_weights.ptr);
        auto ptr_offsets = static_cast<int *>(bi_offsets.ptr);
        auto ptr_output = static_cast<int *>(output.request().ptr);
        if (ptr_weights == nullptr)
        {
            GenerateRandomWalksImpl(ptr_neighbors, ptr_offsets, ptr_output, n_nodes, n_sims_per_node, n_steps_per_node);
        }
        else
        {
            GenerateRandomWalksImpl(ptr_neighbors, ptr_weights, ptr_offsets, ptr_output, n_nodes, n_sims_per_node, n_steps_per_node);
        }
        return output;
    }

    pybind11::array_t<int> GenerateRandomWalks(
        pybind11::array_t<int> neighbors,
        pybind11::array_t<int> offsets,
        int n_sims_per_node,
        int n_steps_per_node)
    {
        return GenerateRandomWalksWeighted(neighbors, pybind11::array_t<double>(0), offsets, n_sims_per_node, n_steps_per_node);
    }

    

}
