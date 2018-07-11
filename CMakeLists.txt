cmake_minimum_required(VERSION 3.1.0)
project(graphz_cpp_module)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGs} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(third_party/pybind11)

set(SOURCE_DIR "graphz/cpp")
include_directories(SOURCE_DIR)
pybind11_add_module(graphz_cpp_module "${SOURCE_DIR}/sampling.cc" "${SOURCE_DIR}/random_walk.cc" "${SOURCE_DIR}/graphz_cpp_module.cc")
