#include "random_walk.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(graphz_cpp_module, m)
{
    m.doc() = "C++ routines for graphz.";
    m.def("generate_random_walk", &graphz::GenerateRandomWalks, "Generates random walks.");
    m.def("generate_random_walk_weighted", &graphz::GenerateRandomWalksWeighted, "Generates random walks.");
}

