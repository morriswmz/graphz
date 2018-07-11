#include "sampling.h"
#include <vector>
#include <iostream>
#include <stdexcept>

namespace graphz
{
    void CreateAliasTable(size_t size, double *weights, double *p, int *q)
    {
        if (size == 0) return;
        // Vose's Alias Method
        std::vector<size_t> small;
        std::vector<size_t> large;
        for (size_t i = 0;i < size;i++)
        {
            p[i] = weights[i] * size;
            q[i] = -1;
            if (p[i] >= 1)
            {
                large.push_back(i);
            }
            else
            {
                small.push_back(i);
            }
        }
        while (!small.empty() && !large.empty())
        {
            auto l = small.back();
            auto g = large.back();
            large.pop_back();
            small.pop_back();
            q[l] = g;
            p[g] = (p[g] + p[l]) - 1.0;
            if (p[g] >= 1)
            {
                large.push_back(g);
            }
            else
            {
                small.push_back(g);
            }
        }
        for (const auto &i : large) p[i] = 1.0;
        for (const auto &i : small) p[i] = 1.0;
    }

    size_t AliasSample(size_t size, double *p, int *q, std::mt19937 &rng)
    {
        if (size == 1) return 0;
        auto i = std::uniform_int_distribution<int>(0, size - 1)(rng);
        auto prob = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        if (prob <= p[i])
        {
            return i;
        }
        else
        {
            if (q[i] < 0)
            {
                throw std::runtime_error("This can never happen!");
            }
            return q[i];
        }
    }

}
