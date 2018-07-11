#pragma once
#include <random>

namespace graphz
{
    /**
     * Prepares the alias table for fast weighted sampling.
     */
    void CreateAliasTable(size_t size, double *weights, double *p, int *q);

    /**
     * Peforms alias sampling.
     */
    size_t AliasSample(size_t size, double *p, int *q, std::mt19937 &rng);
}
